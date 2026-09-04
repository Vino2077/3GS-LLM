#!/usr/bin/env python3
"""Compare FP, simulated INT8, exported INT8, and optional native C logits."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

import torch
from tokenizers import Tokenizer

from int8_reference import load_exported_model, quantize_model
from model import ModelConfig, ThreeGSModel
from sampling import BANNED_SPECIAL_NAMES, special_token_ids


def top_tokens(logits: torch.Tensor, count: int = 10) -> list[dict[str, float | int]]:
    values, indices = torch.topk(logits.float(), count)
    return [
        {"id": int(identifier), "logit": float(value)}
        for value, identifier in zip(values.cpu(), indices.cpu())
    ]


def differences(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    delta = reference.float() - candidate.float()
    cosine = torch.nn.functional.cosine_similarity(
        reference.float(), candidate.float(), dim=0
    )
    reference_top = {int(value) for value in torch.topk(reference, 10).indices}
    candidate_top = {int(value) for value in torch.topk(candidate, 10).indices}
    return {
        "maximum_absolute": float(delta.abs().max()),
        "mean_absolute": float(delta.abs().mean()),
        "cosine": float(cosine),
        "top10_overlap": len(reference_top & candidate_top) / 10.0,
    }


def native_trace(
    executable: Path | None,
    wasm: Path | None,
    node: Path | None,
    model: Path,
    prefix: list[int],
    forced: list[int],
) -> list[dict[str, object]]:
    arguments = [
        str(model),
        ",".join(map(str, prefix)),
        ",".join(map(str, forced)),
    ]
    if wasm is not None:
        if node is None:
            raise ValueError("--node is required with --native-wasm")
        runner = Path(__file__).resolve().parent.parent / "Scripts" / "run_wasi_trace.mjs"
        command = [str(node), str(runner), str(wasm), *arguments]
    elif executable is not None:
        command = [str(executable), *arguments]
    else:
        raise ValueError("native trace backend was not supplied")
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [json.loads(line) for line in completed.stdout.splitlines() if line]


@torch.inference_mode()
def trace_prompt(
    prompt: str,
    tokenizer: Tokenizer,
    models: dict[str, ThreeGSModel],
    steps: int,
    native_executable: Path | None,
    native_wasm: Path | None,
    node: Path | None,
    model_bin: Path,
) -> dict[str, object]:
    ids = special_token_ids(tokenizer)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    prefix = [ids["bos"], ids["user"], *prompt_ids, ids["assistant"]]
    history = prefix[:]
    banned = {ids[name] for name in BANNED_SPECIAL_NAMES}
    records = []
    forced: list[int] = []
    for step in range(steps):
        inputs = torch.tensor(
            [history], dtype=torch.long, device=next(models["fp"].parameters()).device
        )
        logits = {name: model(inputs)[0][0, -1].float() for name, model in models.items()}
        record = {
            "step": step,
            "input_tokens": history[:],
            "fp_top": top_tokens(logits["fp"]),
            "simulated_int8_top": top_tokens(logits["simulated_int8"]),
            "exported_int8_top": top_tokens(logits["exported_int8"]),
            "fp_vs_simulated": differences(logits["fp"], logits["simulated_int8"]),
            "simulated_vs_exported": differences(
                logits["simulated_int8"], logits["exported_int8"]
            ),
        }
        scores = logits["fp"].clone()
        scores[list(banned)] = float("-inf")
        chosen = int(torch.argmax(scores))
        record["forced_next_token"] = chosen
        records.append(record)
        forced.append(chosen)
        history.append(chosen)
    result: dict[str, object] = {
        "prompt": prompt,
        "prefix_tokens": prefix,
        "forced_tokens": forced,
        "steps": records,
    }
    if native_executable is not None or native_wasm is not None:
        native = native_trace(
            native_executable,
            native_wasm,
            node,
            model_bin,
            prefix,
            forced[:-1],
        )
        result["native_c"] = native
        for record, native_record in zip(records, native):
            exported_values = {
                item["id"]: item["logit"] for item in record["exported_int8_top"]
            }
            native_values = {
                item["id"]: item["logit"] for item in native_record["top"]
            }
            shared = exported_values.keys() & native_values.keys()
            shared_differences = [
                abs(float(exported_values[token]) - float(native_values[token]))
                for token in shared
            ]
            record["exported_vs_native"] = {
                "top1_equal": (
                    record["exported_int8_top"][0]["id"]
                    == native_record["top"][0]["id"]
                ),
                "top10_overlap": len(shared) / 10.0,
                "shared_top_maximum_absolute": max(shared_differences),
                "shared_top_mean_absolute": (
                    sum(shared_differences) / len(shared_differences)
                ),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("model_bin", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompts", nargs="+", default=["Скайрим играл?", "Ты кто?"])
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--native-executable", type=Path)
    parser.add_argument("--native-wasm", type=Path)
    parser.add_argument("--node", type=Path)
    args = parser.parse_args()
    if args.native_executable and args.native_wasm:
        parser.error("use only one native backend")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    fp = ThreeGSModel(ModelConfig(**checkpoint["config"]))
    fp.load_state_dict(checkpoint["model"])
    simulated = copy.deepcopy(fp)
    quantize_model(simulated)
    models = {
        "fp": fp.to(device).eval(),
        "simulated_int8": simulated.to(device).eval(),
        "exported_int8": load_exported_model(args.model_bin, device),
    }
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    report = {
        "checkpoint": str(args.checkpoint),
        "model_bin": str(args.model_bin),
        "forced_path": "FP greedy tokens are fed to every backend",
        "prompts": [
            trace_prompt(
                prompt,
                tokenizer,
                models,
                args.steps,
                args.native_executable,
                args.native_wasm,
                args.node,
                args.model_bin,
            )
            for prompt in args.prompts
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote runtime comparison -> {args.output}")


if __name__ == "__main__":
    main()
