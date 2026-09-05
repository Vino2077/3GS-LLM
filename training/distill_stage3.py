"""Durable candidate generation + separately shuffled judge pass; rejected rows stay auditable."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time

from prepare_sft import records
from stage3_quality import response_rejection
from teacher_client import TeacherClient, TeacherOutputError, GENERATION_PROMPT, JUDGE_PROMPT, validate_judgment, acceptable


def append_durable(file, row):
    file.write(json.dumps(row, ensure_ascii=False) + "\n")
    file.flush()
    os.fsync(file.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--model", default="3gs-teacher-qwen35")
    ap.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--lmstudio-native", action="store_true")
    ap.add_argument("--cooldown", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260905)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = {"model": a.model, "base_url": a.base_url, "lmstudio_native": a.lmstudio_native,
        "input_sha256": hashlib.sha256(a.input.read_bytes()).hexdigest(), "seed": a.seed,
        "generation_prompt": GENERATION_PROMPT, "judge_prompt": JUDGE_PROMPT}
    manifest_path = a.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Manifest mismatch: use a new output directory")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path = a.output / "decisions.jsonl"
    previous = list(records(log_path)) if log_path.exists() else []
    done = {row["id"] for row in previous}
    stats = Counter(row["status"] for row in previous)
    client = TeacherClient(a.base_url, a.model, lmstudio_native=a.lmstudio_native)
    started = time.monotonic()
    processed = 0
    with log_path.open("a", encoding="utf-8") as log:
        for index, item in enumerate(records(a.input)):
            if index >= a.limit:
                break
            identity = item.get("id") or hashlib.sha256(item["prompt"].encode()).hexdigest()
            if identity in done:
                continue
            row = {"id": identity, "prompt": item["prompt"], "source": item.get("source", "dtf")}
            try:
                generation = client.chat(GENERATION_PROMPT, {"prompt": item["prompt"]}, a.seed + index,
                                         0.65, 420)
            except TeacherOutputError as error:
                row.update(status="rejected_generation_format", error=str(error), raw_output=error.raw)
                append_durable(log, row)
                done.add(identity)
                stats[row["status"]] += 1
                processed += 1
                continue
            row["generation"] = generation
            data = generation["data"]
            if type(data.get("self_contained")) is not bool or not isinstance(data.get("candidates"), list):
                row["status"] = "rejected_generation_schema"
                append_durable(log, row)
                done.add(identity)
                stats[row["status"]] += 1
                processed += 1
                continue
            if not data["self_contained"]:
                row["status"] = "rejected_context"
            else:
                candidates = data["candidates"]
                if len(candidates) != 2 or any(not isinstance(c, str) for c in candidates):
                    row["status"] = "rejected_generation_schema"
                    append_durable(log, row)
                    done.add(identity)
                    stats[row["status"]] += 1
                    processed += 1
                    continue
                rng = random.Random(a.seed + index)
                candidates = candidates.copy()
                rng.shuffle(candidates)
                try:
                    judgment = client.chat(JUDGE_PROMPT, {"prompt": item["prompt"], "candidates": candidates},
                                           a.seed + 100000 + index, 0.0, 650)
                    validate_judgment(judgment["data"], len(candidates))
                except (TeacherOutputError, ValueError) as error:
                    row.update(status="rejected_judge_format", error=str(error), raw_output=getattr(error, "raw", None))
                    append_durable(log, row)
                    done.add(identity)
                    stats[row["status"]] += 1
                    processed += 1
                    continue
                row["judgment"] = judgment
                row["judged_candidates"] = candidates
                keep = []
                for candidate, score in zip(candidates, judgment["data"]["scores"]):
                    if (judgment["data"]["self_contained"] and acceptable(score) and
                            response_rejection(item["prompt"], candidate) is None):
                        keep.append((score["relevance"] * 5 + score["directness"] * 3 + score["fluency"] + score["style"], candidate))
                row["status"] = "accepted" if keep else "rejected_judge"
                if keep:
                    row["response"] = max(keep, key=lambda x: (x[0], -len(x[1])))[1]
            append_durable(log, row)
            done.add(identity)
            stats[row["status"]] += 1
            processed += 1
            if processed % 10 == 0:
                print(json.dumps({"processed_this_run": processed, "stats": dict(stats),
                    "seconds_per_prompt": round((time.monotonic() - started) / processed, 2)}, ensure_ascii=False), flush=True)
            if a.cooldown:
                time.sleep(a.cooldown)
    # Rebuild accepted corpus from authoritative durable decisions, including resumed work.
    with (a.output / "accepted.jsonl").open("w", encoding="utf-8") as out:
        for row in records(log_path):
            if row["status"] == "accepted":
                out.write(json.dumps({"id": row["id"], "prompt": row["prompt"], "response": row["response"],
                    "source": "teacher-distillation", "prompt_source": row["source"]}, ensure_ascii=False) + "\n")
    report = {"stats": dict(stats), "seconds_this_run": time.monotonic() - started,
              "processed_this_run": processed, "note": "Local judge acceptance still requires human audit."}
    (a.output / "stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
