#!/usr/bin/env python3
"""Inject private model assets into an already signed jailbreak IPA shell."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

from verify_model_assets import verify_model, verify_tokenizer


def asset_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shell_ipa", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    verify_model(args.model)
    verify_tokenizer(args.tokenizer)
    if args.output.resolve() == args.shell_ipa.resolve():
        raise ValueError("output IPA must differ from the shell IPA")

    with zipfile.ZipFile(args.shell_ipa, "r") as source:
        app_roots = sorted(
            name for name in source.namelist()
            if name.startswith("Payload/") and name.endswith(".app/")
            and name.count("/") == 2
        )
        if len(app_roots) != 1:
            raise ValueError(f"expected one app bundle, found {app_roots}")
        app_root = app_roots[0]
        if any(
            name.startswith(app_root + "_CodeSignature/")
            or name == app_root + "CodeResources"
            for name in source.namelist()
        ):
            raise ValueError("IPA has a resource seal and cannot be safely injected")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="3gs-llm-", suffix=".ipa", dir=args.output.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        try:
            with zipfile.ZipFile(temporary_path, "w", allowZip64=False) as target:
                for info in source.infolist():
                    if info.filename in {
                        app_root + "model.bin",
                        app_root + "tokenizer.bin",
                    }:
                        continue
                    target.writestr(info, source.read(info.filename))
                target.writestr(
                    asset_info(app_root + "model.bin"), args.model.read_bytes()
                )
                target.writestr(
                    asset_info(app_root + "tokenizer.bin"),
                    args.tokenizer.read_bytes(),
                )
            shutil.move(str(temporary_path), args.output)
        finally:
            temporary_path.unlink(missing_ok=True)

    with zipfile.ZipFile(args.output, "r") as result:
        if result.getinfo(app_root + "model.bin").file_size != args.model.stat().st_size:
            raise ValueError("injected model size mismatch")
        if result.getinfo(app_root + "tokenizer.bin").file_size != args.tokenizer.stat().st_size:
            raise ValueError("injected tokenizer size mismatch")
    print(f"created {args.output} ({args.output.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
