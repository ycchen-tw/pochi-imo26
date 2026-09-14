#!/usr/bin/env python3
"""Extract the pinned public Pochi runtime without a Docker daemon."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time
import urllib.request

base = Path(sys.argv[1]).resolve()
run = Path(sys.argv[2]).resolve()
runtime = base / "runtime"
local_runtime = Path(
    os.environ.get("FM_POCHI_RUNTIME_STORAGE", str(Path(__file__).resolve().parent / "runtime"))
)
archive = run / "pp-runtime-layer.tar.gz"
digest = "27c911493f490231f95909cb831ce7d958cd5f2604968dedde7930744708c130"
size = 4673890422
registry = "https://ghcr.io/v2/fieldsmodelorg/aimo-proof-pilot/"
os.umask(0o002)


def status(phase, **kw):
    (run / "bootstrap-status.json").write_text(json.dumps({"phase": phase, **kw}, indent=2) + "\n")
    print(phase, kw, flush=True)


if not archive.exists():
    status("downloading_runtime", total_bytes=size)
    url = "https://ghcr.io/token?service=ghcr.io&scope=repository:fieldsmodelorg/aimo-proof-pilot:pull"
    with urllib.request.urlopen(url, timeout=60) as r:
        token = json.load(r)["token"]
    req = urllib.request.Request(registry + "blobs/sha256:" + digest,
                                 headers={"Authorization": "Bearer " + token})
    partial = archive.with_suffix(".incomplete")
    started = last = time.monotonic()
    total = 0
    with urllib.request.urlopen(req, timeout=120) as response, partial.open("wb") as out:
        while chunk := response.read(8 * 1024 * 1024):
            out.write(chunk)
            total += len(chunk)
            if time.monotonic() - last > 20:
                status("downloading_runtime", bytes=total, total_bytes=size)
                last = time.monotonic()
    if total != size:
        raise RuntimeError(f"Wrong runtime archive size: {total}")
    partial.rename(archive)

status("verifying_runtime_archive")
sha = hashlib.sha256()
with archive.open("rb") as stream:
    for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
        sha.update(chunk)
assert sha.hexdigest() == digest, "Runtime layer SHA256 mismatch"

if not (runtime / ".extracted.json").exists():
    stage = run / "runtime-stage"
    stage.mkdir(exist_ok=True)
    status("extracting_runtime")
    count = 0
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            parts = PurePosixPath(member.name).parts
            if parts[:2] != ("opt", "pp") or len(parts) <= 2:
                continue
            member.name = str(PurePosixPath(*parts[2:]))
            if member.issym() and member.linkname.startswith("/opt/pp/"):
                target = stage / member.linkname[len("/opt/pp/"):]
                member.linkname = os.path.relpath(target, (stage / member.name).parent)
            elif member.islnk() and member.linkname.startswith("opt/pp/"):
                member.linkname = member.linkname[len("opt/pp/"):]
            tar.extract(member, path=stage, filter="data")
            count += 1
            if count % 10000 == 0:
                status("extracting_runtime", files=count)
    status("installing_runtime_on_local_storage", files=count)
    local_runtime.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(local_runtime)
    runtime.symlink_to(local_runtime, target_is_directory=True)
    (runtime / ".extracted.json").write_text(json.dumps({"layer_sha256": digest, "files": count}) + "\n")

status("relocating_runtime")
venv = runtime / "venv"
pybase = runtime / "pybase"
cfg = venv / "pyvenv.cfg"
cfg.write_text("\n".join("home = " + str(pybase / "bin") if l.startswith("home =") else l
                         for l in cfg.read_text().splitlines()) + "\n")
for p in (venv / "bin").iterdir():
    if p.is_symlink():
        link = os.readlink(p)
        if link.startswith("/opt/pp/"):
            p.unlink()
            p.symlink_to(os.path.relpath(runtime / link[len("/opt/pp/"):], p.parent))
        continue
    if p.is_file():
        with p.open("rb") as f:
            first = f.readline(4096)
        if first.startswith(b"#!") and b"python" in first:
            data = p.read_bytes()
            p.write_bytes(("#!" + str(venv / "bin/python3") + "\n").encode() + data[len(first):])
status("extracted_and_relocated", runtime=str(runtime), layer_sha256=digest)
