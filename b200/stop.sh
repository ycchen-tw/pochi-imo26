#!/usr/bin/env bash
set -Eeuo pipefail
CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$CODE_DIR/env.sh"
"$VENV/bin/python" - <<'PY'
import json, os, signal, time
from pathlib import Path
file = Path(os.environ['FM_POCHI_ROOT'],'service.json')
state = json.loads(file.read_text())
pid = state['pid']
proc = Path(f'/proc/{pid}')
if not proc.exists():
    print('Server is already stopped.')
    raise SystemExit(0)
if proc.joinpath('stat').read_text().split()[21] != state['start_ticks']:
    raise SystemExit('PID has been reused; refusing to stop it.')
if 'sglang.launch_server' not in proc.joinpath('cmdline').read_text():
    raise SystemExit('PID is not the recorded SGLang server; refusing to stop it.')
if os.getpgid(pid) != pid:
    raise SystemExit('Unexpected process group; refusing to stop it.')
os.killpg(pid, signal.SIGTERM)
print(f'Sent SIGTERM to Pochi process group {pid}.')
PY
