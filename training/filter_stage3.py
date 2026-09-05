"""Stream every training row, deduplicate parents and shortlist for semantic review."""
import argparse
from collections import Counter
import hashlib
import json
import random
from pathlib import Path

from prepare_sft import records
from stage3_quality import normalized, prompt_rejection, response_rejection, near_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--exclude", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=20260905)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    excluded = [normalized(x["prompt"]) for x in json.loads(a.exclude.read_text(encoding="utf-8"))]
    seen, pool, stats = set(), [], Counter()
    rng = random.Random(a.seed)
    with (a.output / "original_candidates.jsonl").open("w", encoding="utf-8") as originals:
        for item in records(a.input):
            stats["rows_considered"] += 1
            p = str(item["parent"]).strip()
            identity = normalized(p)
            reason = prompt_rejection(p)
            if reason:
                stats["rejected_" + reason] += 1
                continue
            if near_eval(p, excluded):
                stats["rejected_eval_overlap"] += 1
                continue
            likes = int(item.get("response_likes", 0))
            if likes >= 10 and not response_rejection(p, str(item["response"])):
                originals.write(json.dumps({"prompt": p, "response": item["response"],
                    "likes": likes, "source": "original-dtf"}, ensure_ascii=False) + "\n")
                stats["original_candidates"] += 1
            if identity in seen:
                stats["duplicate_parents"] += 1
                continue
            seen.add(identity)
            stats["heuristic_eligible_unique"] += 1
            row = {"id": "dtf-" + hashlib.sha256(identity.encode()).hexdigest()[:24],
                   "prompt": p, "source": "dtf", "source_response_likes": likes}
            if len(pool) < a.limit:
                pool.append(row)
            else:
                index = rng.randrange(stats["heuristic_eligible_unique"])
                if index < a.limit:
                    pool[index] = row
    rng.shuffle(pool)
    with (a.output / "teacher_prompts.jsonl").open("w", encoding="utf-8") as out:
        for row in pool:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats["shortlisted"] = len(pool)
    report = {"seed": a.seed, "counts": dict(stats),
              "note": "Heuristic eligibility is not confirmed self-containedness. Teacher/judge review required."}
    (a.output / "filter_stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
