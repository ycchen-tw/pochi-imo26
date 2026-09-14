#!/usr/bin/env python3
"""Supervise this bounded FP8/direct campaign and release only its owned server."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_direct import write_json


def ticks(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[21]
    except FileNotFoundError:
        return None


def stop_owned(pid, start_ticks, marker):
    if start_ticks is None or ticks(pid) != start_ticks:
        return
    command = Path(f'/proc/{pid}/cmdline').read_text()
    if marker not in command or os.getpgid(pid) != pid:
        raise RuntimeError(f'refusing cleanup of unexpected PID {pid}')
    os.killpg(pid, signal.SIGTERM)
    for _ in range(30):
        if ticks(pid) != start_ticks:
            return
        time.sleep(1)
    if ticks(pid) == start_ticks:
        os.killpg(pid, signal.SIGKILL)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', required=True, type=Path)
    ap.add_argument('--max-seconds', type=int, default=14400)
    args = ap.parse_args()
    root = args.run_dir.resolve()
    service = json.loads((root/'serve/service.json').read_text())
    assert ticks(service['pid']) == service['start_ticks'], 'recorded server is not live'
    info = json.loads((root/'server-info.json').read_text())
    assert info['quantization'] == 'fp8' and info['kv_cache_dtype'] == 'fp8_e4m3'
    assert json.loads((root/'smoke.json').read_text())['passed'], 'smoke must pass first'
    child = None
    child_ticks = None
    status = {'state':'running', 'started_at':datetime.now(timezone.utc).isoformat(),
              'pid':os.getpid(), 'start_ticks':ticks(os.getpid()), 'server_pid':service['pid']}
    def save():
        write_json(root/'campaign-status.json', status)
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    save()
    try:
        with (root/'direct.log').open('a') as log:
            command = [sys.executable, str(ROOT/'run_direct.py'), '--run-dir', str(root/'direct'),
                       '--k','10','--concurrency','100','--max-tokens','131072']
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            child_ticks = ticks(child.pid)
            status.update(runner_pid=child.pid, runner_start_ticks=child_ticks, command=command)
            save()
            started = time.monotonic()
            while child.poll() is None:
                if time.monotonic()-started > args.max_seconds:
                    raise TimeoutError('campaign wall-clock budget exhausted')
                if ticks(service['pid']) != service['start_ticks']:
                    raise RuntimeError('owned FP8 server exited during benchmark')
                progress = root/'direct/status.json'
                if progress.exists():
                    status['progress'] = json.loads(progress.read_text())
                status['updated_at'] = datetime.now(timezone.utc).isoformat()
                save()
                time.sleep(10)
            if child.returncode:
                raise RuntimeError(f'direct runner exited {child.returncode}')
        summary=json.loads((root/'direct/summary.json').read_text())
        assert summary['complete'] and summary['completed_samples']==100 and summary['artifact_audit_passed']
        status.update(state='completed', pass_at_10=summary['pass_at_k'], correct_samples=summary['correct_samples'])
    except BaseException as error:
        status.update(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed', error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            stop_owned(child.pid, child_ticks, 'run_direct.py')
            child.wait(timeout=15)
        stop_owned(service['pid'], service['start_ticks'], 'sglang.launch_server')
        status.update(finished_at=datetime.now(timezone.utc).isoformat(), server_released=ticks(service['pid']) != service['start_ticks'])
        save()
        gpu=subprocess.check_output(['nvidia-smi'],text=True)
        (root/'gpu-after.txt').write_text(gpu)
        write_json(root/'PAUSED.json', {'state':'stopped_after_validation', 'server_pid':service['pid'], 'campaign_state':status['state'], 'checked_at':status['finished_at']})


if __name__=='__main__':
    main()
