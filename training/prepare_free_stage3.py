"""Build transparent mixtures and strict short-DTF candidates, without fabricated teacher counts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
from prepare_sft import records
from stage3_quality import normalized, near_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("authored", type=Path)
    ap.add_argument("filtered", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--exclude", type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    excluded = [normalized(x['prompt']) for x in json.loads(a.exclude.read_text(encoding='utf-8'))]
    rows = list(records(a.authored))
    simple = [r for r in rows if r['source'] != 'authored-game-alignment']
    # Oversampling is intentional and reported as such, not new examples.
    mixtures = {'A': rows + simple * 3, 'simple': simple}
    for name, corpus in mixtures.items():
        random.Random(20260905).shuffle(corpus)
        with (a.output / f'{name}.jsonl').open('w', encoding='utf-8') as out:
            for row in corpus:
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
    topic = re.compile(r'игр|гейм|скайрим|ведьмак|киберпанк|дота|doom|skyrim|фильм|кино|сериал|консол|айфон|телефон|наушник|видеокарт|ремейк|график|сюжет|steam|ps5|ps4|xbox|мультик|аниме|персонаж|компьютер|ноутбук', re.I)
    hidden = re.compile(r'\b(он|она|они|эти|эта|этот|этого|этой|этом|этим|такое|такие|там|тут|выше|ниже|слева|справа|написано|писал|пост|автор|спойлернул)\b', re.I)
    candidates, original, seen = [], [], set()
    for row in records(a.filtered / 'teacher_prompts.jsonl'):
        p = row['prompt']
        if 12 <= len(p) <= 150 and '\n' not in p and topic.search(p) and not hidden.search(p) and not near_eval(p, excluded):
            candidates.append(row)
    for row in records(a.filtered / 'original_candidates.jsonl'):
        p, r = row['prompt'], row['response']
        key = normalized(p)
        if key in seen:
            continue
        if (12 <= len(p) <= 150 and len(r) <= 180 and '\n' not in p and '\n' not in r
            and topic.search(p) and not hidden.search(p) and not near_eval(p, excluded)
            and row['likes'] >= 25):
            seen.add(key)
            original.append(row)
    rng = random.Random(20260905)
    rng.shuffle(candidates)
    rng.shuffle(original)
    for name, corpus in [('strict_prompts', candidates), ('strict_originals', original)]:
        with (a.output / f'{name}.jsonl').open('w', encoding='utf-8') as out:
            for row in corpus:
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({'authored_unique_pairs':len(rows), 'simple_unique_pairs':len(simple),
                      'strict_prompt_candidates':len(candidates), 'strict_original_candidates':len(original)}, indent=2))


if __name__ == '__main__':
    main()
