from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Final


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write(f"usage: {Path(sys.argv[0]).name} WHEEL\n")
        return 2

    wheel: Final = Path(sys.argv[1])
    with zipfile.ZipFile(wheel) as archive:
        native_members: Final = tuple(
            member
            for member in archive.infolist()
            if member.filename.startswith("litellm/rust_bridge/_native.") and member.filename.endswith(".so")
        )
        if len(native_members) != 1:
            sys.stderr.write(f"expected one native extension, found {len(native_members)}\n")
            return 1

        native_member: Final = native_members[0]
        uncompressed_wheel_size: Final = sum(member.file_size for member in archive.infolist())
        native_path: Final = wheel.parent / "native" / Path(native_member.filename).name
        native_path.parent.mkdir(parents=True, exist_ok=True)
        native_path.write_bytes(archive.read(native_member))

    report: Final = "\n".join(
        (
            "## Release wheel size",
            "",
            "| Artifact | Bytes |",
            "| --- | ---: |",
            f"| Compressed wheel | {wheel.stat().st_size:,} |",
            f"| Uncompressed wheel | {uncompressed_wheel_size:,} |",
            f"| Native extension | {native_member.file_size:,} |",
            "",
        )
    )
    summary_path: Final = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path is None:
        sys.stdout.write(report)
    else:
        Path(summary_path).write_text(report)

    sections: Final = subprocess.run(
        ("readelf", "--sections", "--wide", native_path),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    forbidden_sections: Final = tuple(
        section for section in (".symtab", ".debug_", ".zdebug_") if section in sections
    )
    if forbidden_sections:
        sys.stderr.write(
            f"{native_member.filename} contains unstripped sections: {', '.join(forbidden_sections)}\n"
        )
        return 1

    dynamic_symbols: Final = subprocess.run(
        ("readelf", "--dyn-syms", "--wide", native_path),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "PyInit__native" not in dynamic_symbols:
        sys.stderr.write("native extension does not export PyInit__native\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
