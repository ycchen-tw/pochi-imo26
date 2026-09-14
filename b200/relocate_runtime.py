#!/usr/bin/env python3
"""Relocate the extracted runtime to its current local path, idempotently."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys
from datetime import datetime, timezone


def atomic_write(path: Path, data: bytes, mode: int | None = None) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    if mode is not None:
        temporary.chmod(mode)
    os.replace(temporary, path)


def relocate_runtime(path: Path) -> dict:
    runtime = path.resolve()
    venv = runtime / "venv"
    pybase = runtime / "pybase"
    for required in (venv / "pyvenv.cfg", venv / "bin", pybase / "bin/python3"):
        if not required.exists():
            raise FileNotFoundError(required)

    cfg = venv / "pyvenv.cfg"
    lines = cfg.read_text().splitlines()
    expected_home = f"home = {pybase / 'bin'}"
    old_home = next((line for line in lines if line.startswith("home =")), None)
    rewritten = [expected_home if line.startswith("home =") else line for line in lines]
    if old_home is None:
        rewritten.insert(0, expected_home)
    cfg_changed = rewritten != lines
    if cfg_changed:
        atomic_write(cfg, ("\n".join(rewritten) + "\n").encode(), stat.S_IMODE(cfg.stat().st_mode))

    rewritten_scripts: list[str] = []
    rewritten_symlinks: list[str] = []
    python_shebang = (f"#!{venv / 'bin/python3'}\n").encode()
    for entry in sorted((venv / "bin").iterdir()):
        if entry.is_symlink():
            target = os.readlink(entry)
            for prefix in ("/opt/pp/", "/nfs/aimo/shared/fm-pochi/runtime/"):
                if target.startswith(prefix):
                    relative = target[len(prefix):]
                    entry.unlink()
                    entry.symlink_to(os.path.relpath(runtime / relative, entry.parent))
                    rewritten_symlinks.append(entry.name)
                    break
            continue
        if not entry.is_file():
            continue
        with entry.open("rb") as stream:
            first = stream.readline(4096)
        if not (first.startswith(b"#!") and b"python" in first):
            continue
        if first == python_shebang:
            continue
        data = entry.read_bytes()
        atomic_write(
            entry,
            python_shebang + data[len(first):],
            stat.S_IMODE(entry.stat().st_mode),
        )
        rewritten_scripts.append(entry.name)

    stale: list[str] = []
    for entry in [cfg, *(venv / "bin").iterdir()]:
        if entry.is_symlink():
            value = os.readlink(entry)
            if value.startswith(("/opt/pp/", "/nfs/aimo/shared/fm-pochi/runtime/")):
                stale.append(str(entry))
        elif entry.is_file():
            with entry.open("rb") as stream:
                first = stream.readline(4096)
            if b"/nfs/aimo/shared/fm-pochi/runtime" in first or b"/opt/pp/" in first:
                stale.append(str(entry))
    if stale:
        raise RuntimeError(f"stale runtime entrypoints remain: {stale}")

    report = {
        "schema_version": 1,
        "relocated_at": datetime.now(timezone.utc).isoformat(),
        "runtime": str(runtime),
        "old_home": old_home,
        "home": expected_home,
        "pyvenv_cfg_changed": cfg_changed,
        "rewritten_scripts": rewritten_scripts,
        "rewritten_symlinks": rewritten_symlinks,
    }
    atomic_write(
        runtime / "RELOCATION.json",
        (json.dumps(report, indent=2) + "\n").encode(),
    )
    return report


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} RUNTIME")
    print(json.dumps(relocate_runtime(Path(sys.argv[1])), indent=2))


if __name__ == "__main__":
    main()
