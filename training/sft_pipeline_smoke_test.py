#!/usr/bin/env python3
"""Exercise SFT validation, aligned packing, and teacher interchange formats."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def run(*arguments: object, working_directory: Path) -> None:
    subprocess.run(
        [sys.executable, *map(str, arguments)],
        cwd=working_directory,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", type=Path)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="3gs-sft-") as temporary:
        root = Path(temporary)
        input_path = root / "sft.jsonl"
        with input_path.open("w", encoding="utf-8", newline="\n") as target:
            for index in range(400):
                target.write(
                    json.dumps(
                        {
                            "prompt": f"Как проверить выровненную пару номер {index}?",
                            "response": f"Ответ номер {index} начинается после assistant.",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        prepared = root / "prepared"
        packed = root / "packed"
        run("training/prepare_sft.py", input_path, prepared, working_directory=repository)
        run(
            "training/tokenize_dataset.py",
            prepared,
            args.tokenizer.resolve(),
            packed,
            working_directory=repository,
        )
        for split in ("train", "validation"):
            tokens = np.fromfile(packed / f"{split}.bin", dtype="<u2")
            masks = np.fromfile(packed / f"{split}_response_mask.bin", dtype="u1")
            offsets = np.fromfile(packed / f"{split}_offsets.bin", dtype="<u8")
            assert len(tokens) == len(masks) and offsets[-1] == len(tokens)
            for start, end in zip(offsets[:-1], offsets[1:]):
                sample = tokens[int(start) : int(end)]
                mask = masks[int(start) : int(end)]
                assistant = int(np.flatnonzero(sample == 4)[0])
                assert sample[0] == 1 and sample[1] == 3 and sample[-1] == 2
                assert not mask[: assistant + 1].any()
                assert mask[assistant + 1 :].all()

        prompts = root / "distillation-prompts.jsonl"
        run(
            "training/build_distillation_prompts.py",
            prepared / "train.jsonl.gz",
            prompts,
            "--count",
            20,
            "--allow-small",
            working_directory=repository,
        )
        requests = root / "teacher-requests.jsonl"
        run(
            "training/make_teacher_requests.py",
            prompts,
            "training/teacher_prompt.txt",
            requests,
            working_directory=repository,
        )
        teacher_outputs = root / "teacher-outputs.jsonl"
        with prompts.open("r", encoding="utf-8") as source, teacher_outputs.open(
            "w", encoding="utf-8", newline="\n"
        ) as target:
            for line in source:
                item = json.loads(line)
                target.write(
                    json.dumps(
                        {
                            "id": item["id"],
                            "prompt": item["prompt"],
                            "response": "Короткий, прямой и проверяемый ответ.",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        imported = root / "imported.jsonl"
        run(
            "training/import_teacher_outputs.py",
            teacher_outputs,
            imported,
            working_directory=repository,
        )
        assert sum(1 for _ in imported.open(encoding="utf-8")) == 20
        with gzip.open(prepared / "train.jsonl.gz", "rt", encoding="utf-8") as source:
            assert sum(1 for _ in source) > 300
        print("SFT/distillation pipeline smoke test passed")


if __name__ == "__main__":
    main()
