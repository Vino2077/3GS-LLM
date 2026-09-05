"""Shared, conservative Stage 3 data checks; heuristics are NOT semantic judges."""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import lru_cache


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[\w]+", text.casefold().replace("ё", "е")))


def copy_metrics(prompt: str, response: str) -> dict:
    p, r = normalized(prompt), normalized(response)
    pw, rw = p.split(), r.split()
    match = SequenceMatcher(None, pw, rw, autojunk=False).find_longest_match()
    exact = bool(p) and p == r
    prefix = bool(p) and (r == p or r.startswith(p + " "))
    # Single entity mentions and normal two-word direct answers are not echoes.
    overlap = match.size / max(1, len(pw))
    copying = exact or (prefix and len(pw) >= 2) or (
        len(pw) >= 5 and match.size >= 5 and overlap >= 0.8
    )
    return {"prompt_copy": copying, "exact_echo": exact, "prefix_echo": prefix,
            "longest_prompt_word_overlap": overlap}


def prompt_rejection(prompt: str) -> str | None:
    text = normalized(prompt)
    words = text.split()
    if not 8 <= len(prompt) <= 420 or not 2 <= len(words) <= 75:
        return "length"
    if len(re.findall(r"[а-яё]", prompt.lower())) < 4:
        return "not_russian"
    if re.search(r"https?://|@|<\|", prompt):
        return "link_or_markup"
    if re.search(r"(?:на|в) (?:фото|картинке|скрине|скриншоте)|слева|справа|выше (?:же|напис)|ниже (?:же|напис)|автор (?:поста|статьи)|в (?:посте|статье)|этот коммент|на второй|на первой|тот случай", text):
        return "hidden_context"
    if re.match(r"^(вот именно|и этот|и эта|и он|и она|он опять|она опять|это после|ну выше|тот самый)( |$)", text):
        return "deictic_fragment"
    if len(words) <= 7 and re.match(r"^(он|она|они|этот|эта|эти|там|тут|это тоже|так и)( |$)", text):
        return "short_reference"
    return None


def response_rejection(prompt: str, response: str) -> str | None:
    text = normalized(response)
    if not 5 <= len(response) <= 320:
        return "response_length"
    if len(re.findall(r"[а-яё]", response.lower())) < 4:
        return "not_russian"
    if re.match(r"^(конечно|давайте разберемся|как ии|как искусственный интеллект|в качестве ии)", text):
        return "boilerplate"
    if copy_metrics(prompt, response)["prompt_copy"]:
        return "prompt_copy"
    words = text.split()
    triples = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    if triples and 1 - len(set(triples)) / len(triples) > 0.1:
        return "repetition"
    if len(re.findall(r"[.!?]+(?:\s|$)", response)) > 3:
        return "too_many_sentences"
    return None


@lru_cache(maxsize=1024)
def word_set(value: str) -> frozenset:
    return frozenset(value.split())


def near_eval(prompt: str, excluded: list[str]) -> bool:
    value = normalized(prompt)
    words = set(value.split())
    for other in excluded:
        if value == other:
            return True
        if abs(len(value) - len(other)) > max(8, len(value) // 4):
            continue
        # Require lexical overlap before expensive fuzzy matching. Exact normalized
        # matches are always caught above, including punctuation/case differences.
        if len(words & word_set(other)) < min(len(words), len(word_set(other))) * 0.5:
            continue
        if SequenceMatcher(None, value, other).ratio() >= 0.88:
            return True
    return False
