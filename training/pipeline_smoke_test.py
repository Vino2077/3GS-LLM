#!/usr/bin/env python3
"""Exercise cleaning, tokenizer training, and binary packing on synthetic rows."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from prepare_dtf import COLUMNS, split_is_validation


def title_for(validation: bool) -> str:
    for index in range(10_000):
        candidate = f"Synthetic post {index}"
        if split_is_validation(candidate) is validation:
            return candidate
    raise AssertionError("could not construct deterministic split title")


def run(*arguments: object, working_directory: Path) -> None:
    subprocess.run(
        [sys.executable, *map(str, arguments)],
        cwd=working_directory,
        check=True,
    )


def main() -> None:
    repository = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="3gs-pipeline-") as temporary:
        root = Path(temporary)
        raw = root / "raw"
        data = raw / "data"
        clean = root / "clean"
        tokenizer = root / "tokenizer"
        tokens = root / "tokens"
        data.mkdir(parents=True)
        validation_title = title_for(True)
        training_title = title_for(False)

        next_id = 1
        for shard in range(8):
            rows = []
            for row_index in range(64):
                index = shard * 64 + row_index
                rows.append(
                    {
                        "post_title": validation_title if row_index < 8 else training_title,
                        "parent_comment": (
                            f"Почему старый айфон всё ещё работает номер {index}? "
                            "Это синтетическая строка для проверки корпуса."
                        ),
                        "child_comment": (
                            f"Потому что оптимизация иногда важнее мощности номер {index}."
                        ),
                        "comment_id_parent": next_id,
                        "comment_id_child": next_id + 1,
                        "parent_likes": 10,
                        "child_likes": 7,
                        "reply_count": 2,
                        "parent_comment_tox": 0.05,
                        "child_comment_tox": 0.10,
                    }
                )
                next_id += 2
            table = pa.Table.from_pylist(rows, schema=pa.schema([
                pa.field(name, pa.string()) if name in {
                    "post_title", "parent_comment", "child_comment"
                } else pa.field(name, pa.float64()) if name.endswith("_tox") else pa.field(name, pa.int64())
                for name in COLUMNS
            ]))
            pq.write_table(table, data / f"train-{shard:05d}-of-00008.parquet")

        run("training/prepare_dtf.py", raw, clean, working_directory=repository)
        run(
            "training/train_tokenizer.py",
            clean / "train.jsonl.gz",
            tokenizer,
            "--vocab-size",
            512,
            working_directory=repository,
        )
        run(
            "training/tokenize_dataset.py",
            clean,
            tokenizer / "tokenizer.json",
            tokens,
            working_directory=repository,
        )

        metadata = json.loads((tokens / "dataset_meta.json").read_text())
        for split in ("train", "validation"):
            token_bytes = (tokens / f"{split}.bin").stat().st_size
            mask_bytes = (tokens / f"{split}_response_mask.bin").stat().st_size
            assert token_bytes == mask_bytes * 2
            assert metadata["splits"][split]["pairs"] > 0
        print("pipeline smoke test passed")


if __name__ == "__main__":
    main()
