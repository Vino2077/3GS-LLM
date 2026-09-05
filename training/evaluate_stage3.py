"""Resumable multi-seed evaluation: raw answers first, lexical metrics are not semantics."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer
from model import ModelConfig, ThreeGSModel
from sampling import PRESETS, generate_reply, repeated_ngram_rate
from stage3_quality import copy_metrics


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("tokenizer", type=Path)
    ap.add_argument("prompts", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--presets", nargs="+", choices=list(PRESETS), default=list(PRESETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[20260905, 42, 789])
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = {"checkpoint": str(a.checkpoint.resolve()), "checkpoint_sha256": sha(a.checkpoint),
        "tokenizer_sha256": sha(a.tokenizer), "prompts_sha256": sha(a.prompts),
        "presets": a.presets, "seeds": a.seeds, "max_new_tokens": a.max_new_tokens,
        "device": a.device, "torch": torch.__version__, "generation": "FP32"}
    path = a.output / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Evaluation directory belongs to a different run")
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    results_path = a.output / "responses.jsonl"
    rows = [json.loads(s) for s in results_path.read_text(encoding="utf-8").splitlines()] if results_path.exists() else []
    done = {(r["id"], r["preset"], r["seed"]) for r in rows}
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    model = ThreeGSModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(a.device).eval()
    tokenizer = Tokenizer.from_file(str(a.tokenizer))
    prompts = json.loads(a.prompts.read_text(encoding="utf-8"))
    if len({p["id"] for p in prompts}) != len(prompts):
        raise ValueError("Duplicate eval IDs")
    with results_path.open("a", encoding="utf-8") as out:
        for name in a.presets:
            for item in prompts:
                # Greedy has exactly one result: duplicated deterministic runs give no evidence.
                for seed in a.seeds[:1] if name == "greedy" else a.seeds:
                    if (item["id"], name, seed) in done:
                        continue
                    result = generate_reply(model, tokenizer, item["prompt"], PRESETS[name], seed, a.max_new_tokens)
                    row = {**item, "preset": name, "seed": seed, "response": result.text,
                        "tokens": result.token_ids, "eos": result.reached_eos,
                        "repeated_3gram_rate": repeated_ngram_rate(result.token_ids),
                        **copy_metrics(item["prompt"], result.text)}
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    os.fsync(out.fileno())
                    rows.append(row)
                    if len(rows) % 50 == 0:
                        print(f"saved {len(rows)} responses", flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row["preset"]].append(row)
    summary = {}
    for name, group in groups.items():
        n = len(group)
        summary[name] = {"responses": n, "prompt_copy_rate": sum(r["prompt_copy"] for r in group) / n,
            "repetition": sum(r["repeated_3gram_rate"] for r in group) / n,
            "eos_rate": sum(r["eos"] for r in group) / n,
            "average_tokens": sum(len(r["tokens"]) for r in group) / n}
    (a.output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (a.output / "answers.txt").write_text("\n\n".join(
        f"{r['id']} | {r['preset']} | seed={r['seed']}\nUSER: {r['prompt']}\nMODEL: {r['response']}" for r in rows), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
