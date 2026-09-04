#!/usr/bin/env python3
"""Fail CI unless an IPA contains the exact legacy binary we intend to ship."""

from __future__ import annotations

import plistlib
import struct
import sys
import zipfile
from pathlib import Path


MH_MAGIC = 0xFEEDFACE
CPU_TYPE_ARM = 12
CPU_SUBTYPE_ARM_V7 = 9
LC_VERSION_MIN_IPHONEOS = 0x25
LC_CODE_SIGNATURE = 0x1D


def decode_version(value: int) -> tuple[int, int, int]:
    return (value >> 16 & 0xFF, value >> 8 & 0xFF, value & 0xFF)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} APP.ipa", file=sys.stderr)
        return 2

    ipa_path = Path(sys.argv[1])
    with zipfile.ZipFile(ipa_path) as archive:
        info_paths = [
            name
            for name in archive.namelist()
            if name.startswith("Payload/") and name.endswith(".app/Info.plist")
        ]
        if len(info_paths) != 1:
            raise RuntimeError(f"expected one app Info.plist, found {info_paths}")

        info_path = info_paths[0]
        info = plistlib.loads(archive.read(info_path))
        executable = info["CFBundleExecutable"]
        binary_path = info_path.rsplit("/", 1)[0] + "/" + executable
        binary = archive.read(binary_path)

    if len(binary) < 28:
        raise RuntimeError("Mach-O header is truncated")

    magic, cpu_type, cpu_subtype, _, command_count, _, _ = struct.unpack_from(
        "<7I", binary, 0
    )
    if magic != MH_MAGIC:
        raise RuntimeError(f"expected 32-bit Mach-O, got magic 0x{magic:08x}")
    if cpu_type != CPU_TYPE_ARM or cpu_subtype & 0x00FFFFFF != CPU_SUBTYPE_ARM_V7:
        raise RuntimeError(
            f"expected ARMv7, got cpu={cpu_type} subtype={cpu_subtype & 0x00FFFFFF}"
        )

    offset = 28
    minimum_ios: tuple[int, int, int] | None = None
    has_code_signature = False
    for _ in range(command_count):
        if offset + 8 > len(binary):
            raise RuntimeError("Mach-O load command header is truncated")
        command, command_size = struct.unpack_from("<2I", binary, offset)
        if command_size < 8 or offset + command_size > len(binary):
            raise RuntimeError("invalid Mach-O load command size")

        base_command = command & 0x7FFFFFFF
        if base_command == LC_VERSION_MIN_IPHONEOS:
            minimum_ios = decode_version(struct.unpack_from("<I", binary, offset + 8)[0])
        elif base_command == LC_CODE_SIGNATURE:
            has_code_signature = True
        offset += command_size

    if minimum_ios != (6, 0, 0):
        raise RuntimeError(f"expected deployment target 6.0.0, got {minimum_ios}")
    if not has_code_signature:
        raise RuntimeError("IPA binary has no code signature load command")
    if info.get("MinimumOSVersion") != "6.0":
        raise RuntimeError("Info.plist MinimumOSVersion is not 6.0")

    print(
        f"verified {ipa_path.name}: ARMv7, minimum iOS 6.0.0, "
        f"code signature present, {len(binary)}-byte executable"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
