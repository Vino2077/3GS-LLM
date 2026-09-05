"""Check a teacher/judge on deliberately good and bad answers before trusting scores."""
import argparse
import json
import os
from pathlib import Path
from teacher_client import TeacherClient, GENERATION_PROMPT, JUDGE_PROMPT, validate_judgment, acceptable


CASES = [
    ("Скайрим вечен?", ["Похоже на то. Моды не дают ему уйти на пенсию.",
      "Киберпанк это не игра, а игра, Киберпанк это не игра, а игра.",
      "Скайрим вечен? А что не так?", "Да, нет, игра не игра, вечен не вечен."], [True, False, False, False]),
    ("Стоит покупать новый айфон?", ["Если старый нормально работает, я бы пока сэкономил.",
      "Ведьмак мне тоже понравился.", "Раз твой телефон вчера утонул, бери новый.",
      "Каждый человек сам решает, что ему делать в жизни."], [True, False, False, False]),
    ("Почему люди любят Дарк Соулс?", ["За кайф после победы над боссом, который час размазывал тебя по полу.",
      "Дарк Соулс это игра, в которую играют игроки.", "Да, бывает.",
      "Потому что в игре есть игра, а не игра."], [True, False, False, False]),
    ("Как дела?", ["Нормально, грею старый айфон своими мыслями.",
      "Как дела? Как дела? Как дела?", "У Zelda хорошая физика."], [True, False, False]),
    ("Он опять", ["Да, Тодд опять продаёт Скайрим."], [False]),
    ("А слева?", ["Слева графика лучше."], [False]),
    ("Ты бы купил?", ["Да, PS5 отличная консоль."], [False]),
    ("Как всех родственников перевести в телеграм? Чисто из-за них держу другой мессенджер на телефоне.",
     ["Начни с семейного чата в Телеграме и помоги тем, кому трудно установить приложение.",
      "Держишь Телеграм только ради них? Может, стоит найти альтернативу."], [True, False]),
    ("Купил первую лицензионную игру в Steam и теперь покупаю только лицензии.",
     ["Один раз попробовал удобную библиотеку — и кошелёк уже на крючке.",
      "Проверь активацию в профиле, может, проблема не в лицензии."], [True, False]),
    ("Не понимаю, за что любят эту игру. Обычный проходной шутер.",
     ["Её любят за интересных персонажей и отличный сюжет."], [False]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    ap.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--model", default="3gs-teacher")
    ap.add_argument("--lmstudio-native", action="store_true")
    ap.add_argument("--reasoning", choices=["off", "on"], default="off")
    ap.add_argument("--max-tokens", type=int, default=1100)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    client = TeacherClient(a.base_url, a.model, lmstudio_native=a.lmstudio_native, reasoning=a.reasoning, timeout=600)
    log = a.output / "calibration.jsonl"
    manifest = {"model": a.model, "base_url": a.base_url, "native": a.lmstudio_native,
                "generation_prompt": GENERATION_PROMPT, "judge_prompt": JUDGE_PROMPT,
                "cases": CASES, "reasoning": a.reasoning, "max_tokens": a.max_tokens}
    manifest = json.loads(json.dumps(manifest))
    manifest_path = a.output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Calibration configuration changed; use a new output directory")
    elif log.exists() and log.stat().st_size:
        raise ValueError("Legacy calibration has no manifest; archive it and use a new directory")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    done = {row["id"] for row in map(json.loads, log.read_text(encoding="utf-8").splitlines())} if log.exists() else set()
    with log.open("a", encoding="utf-8") as out:
        for index, (prompt, candidates, expected) in enumerate(CASES):
            identity = f"judge-{index}"
            if identity in done:
                continue
            result = client.chat(JUDGE_PROMPT, {"prompt": prompt, "candidates": candidates},
                                 20260905 + index, 0.3 if a.reasoning == 'on' else 0.0, a.max_tokens)
            validate_judgment(result["data"], len(candidates))
            predicted = [result["data"]["self_contained"] and acceptable(s) for s in result["data"]["scores"]]
            row = {"id": identity, "prompt": prompt, "candidates": candidates, "expected": expected,
                   "predicted": predicted, **result}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
            print(f"{identity}: expected={expected}, predicted={predicted}", flush=True)
        for index, prompt in enumerate(["Скайрим вечен?", "Ласт оф ас 2 хорошая игра?", "Ты кто?",
              "Ведьмак 3 переоценён", "Грангер лучший персонаж", "Скажи что-нибудь как истинный дтфер"]):
            identity = f"generation-{index}"
            if identity in done:
                continue
            result = client.chat(GENERATION_PROMPT, {"prompt": prompt}, 20260915 + index, 0.6, a.max_tokens)
            out.write(json.dumps({"id": identity, "prompt": prompt, **result}, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
            print(identity + ": " + json.dumps(result["data"], ensure_ascii=False), flush=True)
    print(f"Review calibration manually before generation: {log}")


if __name__ == "__main__":
    main()
