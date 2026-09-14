#!/usr/bin/env python3
"""Supervise original FM-Pochi effort presets, independent repeats and audited scoring."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from run_math_harness import extract_answer
from run_direct import write_json
from diagnostics.run_direct_supervised import stop_owned, ticks

REPEATS = {"medium": 4, "high": 2, "xhigh": 1}


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_task(root, problem_id, expected, effort, replicate):
    """Verify the original selected proof and every successful call's prompt linkage."""
    problem = root / 'problems/row-0000'
    final = json.loads((problem/'final.json').read_text())
    answer = json.loads((problem/'answer.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    config = manifest['search']
    assert manifest['selected_ids'] == [problem_id]
    assert config['seed'] == replicate
    from proof_prompts import parse_generation, parse_verification
    calls = {}
    failed = 0
    for line in (problem/'calls.jsonl').read_text().splitlines():
        call = json.loads(line)
        prompt_path = problem/'prompts'/f"{call['prompt_sha256']}.json"
        prompt = json.loads(prompt_path.read_text())
        encoded = json.dumps(prompt, sort_keys=True, ensure_ascii=False).encode()
        assert hashlib.sha256(encoded).hexdigest() == call['prompt_sha256']
        if call.get('error') is not None:
            failed += 1
            continue
        assert call['sample_id'] not in calls
        calls[call['sample_id']] = call
    assert len(calls) == final['calls_completed']
    winner = json.loads((problem/'proofs'/f"{final['selected_proof_id']}.json").read_text())
    assert winner['proof'] == final['final_proof']
    generated = calls[winner['generation_sample_id']]
    assert generated['finish_reason'] == 'stop'
    proof, _, _ = parse_generation(generated['content'], lenient=config['lenient_parsing'])
    assert proof == winner['proof']
    for review in winner['verifications']:
        call = calls[review['sample_id']]
        assert call['verification_disposition'] == 'accepted'
        _, score = parse_verification(call['content'], lenient=config['lenient_parsing'])
        assert score == review['score']
    assert len(winner['verifications']) >= config['min_valid_verifications']
    extracted = extract_answer(final['final_proof'])
    assert answer['id'] == problem_id and answer['answer'] == extracted
    assert answer['selected_proof_id'] == final['selected_proof_id']
    counts = Counter()
    for call in calls.values():
        counts['physical_requests'] += call.get('physical_request_count', 1)
        counts['completion_tokens'] += call.get('completion_tokens') or 0
        counts['prompt_tokens'] += call.get('physical_prompt_tokens') or call.get('prompt_tokens') or 0
        counts['missing_completion_usage'] += call.get('completion_tokens') is None
        counts['salvaged_calls'] += bool(call.get('salvaged'))
        counts['length_recovered_calls'] += bool(call.get('xml_complete_after_length'))
        counts['estimated_usage_calls'] += any(s.get('stream_aborted') for s in call.get('segments', []))
        stage = 'verify' if '/verify/' in call['stage'] else 'select' if call['stage'].endswith('/select') else 'generate'
        counts[f'{stage}_calls'] += 1
    result = {
        'id':problem_id, 'effort':effort, 'replicate':replicate,
        'answer':extracted, 'expected':expected,
        'correct':extracted is not None and extracted == expected,
        'rounds':final['rounds_completed'], 'selected_proof_id':final['selected_proof_id'],
        'final_source':final['final_source'], 'mean_verifier_score':final['mean_verifier_score'],
        'calls':len(calls), 'failed_call_records':failed, **counts,
        'audit_passed':True, 'audited_at':now(), 'run_dir':str(root),
        'final_sha256':file_hash(problem/'final.json'),
        'calls_sha256':file_hash(problem/'calls.jsonl'),
        'manifest_sha256':file_hash(root/'manifest.json'),
    }
    write_json(root/'audit.json', result)
    return result


def report(root, ids, results, state, direct_root):
    arms = {}
    for effort, k in REPEATS.items():
        arm = [r for r in results if r['effort'] == effort]
        rows = []
        for pid in ids:
            rr = [r for r in arm if r['id'] == pid]
            rows.append({'id':pid, 'completed':len(rr), 'correct':sum(r['correct'] for r in rr),
                         'pass_at_k':any(r['correct'] for r in rr) if len(rr)==k else None})
        complete = len(arm)==len(ids)*k and all(r['completed']==k for r in rows)
        arms[effort] = {
            'k':k, 'completed_attempts':len(arm), 'expected_attempts':len(ids)*k,
            'complete':complete, 'correct_attempts':sum(r['correct'] for r in arm),
            'sample_accuracy':sum(r['correct'] for r in arm)/len(arm) if arm else None,
            'pass_at_k':sum(r['correct']>0 for r in rows)/len(ids) if complete else None,
            'completion_tokens':sum(r.get('completion_tokens',0) for r in arm),
            'physical_requests':sum(r.get('physical_requests',0) for r in arm),
            'estimated_usage_calls':sum(r.get('estimated_usage_calls',0) for r in arm),
            'per_problem':rows,
        }
    summary={'state':state, 'updated_at':now(), 'planned_repeats':REPEATS,
             'completed_attempts':len(results), 'expected_attempts':len(ids)*sum(REPEATS.values()),
             'arms':arms, 'results':results}
    write_json(root/'summary.json',summary)
    text=['# AIMO3 FP8 direct and original-harness evaluation','',f'Updated: {summary["updated_at"]}', '',f'Campaign state: **{state}**. All 10 public reference problems are used.', '',
          'FP8 linear weights (online quantization), FP8 E4M3 KV, TP4 x DP2 on 8 B200.',
          'The harness uses the original medium/high/xhigh search presets; only replicate seeds and the integer-answer task contract are adapted.', '',
          '| Mode | k | Completed attempts | Correct attempts | pass@k |', '|---|---:|---:|---:|---:|']
    direct_summary=direct_root/'direct/summary.json'
    if direct_summary.exists():
        d=json.loads(direct_summary.read_text())
        score=f"{100*d['pass_at_k']:.1f}%" if d.get('complete') else 'pending'
        text.append(f"| direct | {d['k']} | {d['completed_samples']}/{d['expected_samples']} | {d['correct_samples']} | {score} |")
    for effort,a in arms.items():
        score=f"{100*a['pass_at_k']:.1f}%" if a['complete'] else 'pending'
        text.append(f"| {effort} | {a['k']} | {a['completed_attempts']}/{a['expected_attempts']} | {a['correct_attempts']} | {score} |")
    text += ['', 'pass@k means at least one correct selected answer among k independent complete attempts. A partial arm has no final pass@k. Different k values are not a controlled comparison of effort alone.', '',
             '| Problem | medium correct/4 | high correct/2 | xhigh correct/1 |', '|---|---:|---:|---:|']
    for pid in ids:
        cells=[]
        for effort,k in REPEATS.items():
            row=next(r for r in arms[effort]['per_problem'] if r['id']==pid)
            cells.append(f"{row['correct']}/{k}" if row['completed']==k else f"pending ({row['completed']}/{k})")
        text.append('| '+pid+' | '+' | '.join(cells)+' |')
    text += ['', 'Full selected solutions, reasoning, reviews, refinements, selector ballots and per-attempt audits are under `tasks/`. `status.json` identifies the current job and any failure. Re-run the supervisor with the same run directory to resume.', '',
             'KV scales use the uncalibrated runtime default 1.0. No matched BF16 comparison was run. Direct uses 131,072 output tokens; the harness preserves its original 128,000 per-call budget and continuation/selector budgets.', '',
             'Fixed issues: FP8 KV uses trtllm_mha instead of the incompatible pinned FA4 path; direct uses async HTTP instead of the legacy 32-thread bottleneck; harness request IDs are namespaced by run/problem to avoid cross-run collisions.', '']
    text.insert(4, f'Artifacts: `{root}`')
    temporary=root/'README.md.tmp'
    temporary.write_text('\n'.join(text))
    os.replace(temporary,root/'README.md')
    results_path=ROOT/'harness/RESULTS.md'
    temporary=results_path.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(text))
    os.replace(temporary,results_path)
    readme=ROOT/'README.md'
    start='<!-- AIMO3_RESULTS_START -->';end='<!-- AIMO3_RESULTS_END -->'
    content=readme.read_text()
    if start in content and end in content:
        table_start=text.index('| Mode | k | Completed attempts | Correct attempts | pass@k |')
        table_end=next(i for i in range(table_start+2,len(text)) if not text[i])
        block='\n\n'+f"State: **{state}**. Updated: {summary['updated_at']}\n\n"+'\n'.join(text[table_start:table_end])+'\n\n[Per-problem results and validation details](harness/RESULTS.md).\n\n'
        content=content.split(start,1)[0]+start+block+end+content.split(end,1)[1]
        temporary=readme.with_suffix('.md.tmp');temporary.write_text(content);os.replace(temporary,readme)
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', required=True, type=Path)
    ap.add_argument('--direct-run', required=True, type=Path)
    ap.add_argument('--interval', type=int, default=300)
    args=ap.parse_args()
    root=args.run_dir.resolve(); root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=Path(os.environ.get('FM_POCHI_DATA_ROOT', ROOT/'data'))/'aimo3-reference'
    for name in ['problems.csv','reference.csv']:
        if not (root/name).exists():shutil.copy2(source/name,root/name)
        if (root/name).read_bytes()!=(source/name).read_bytes():raise RuntimeError('pinned dataset changed')
    with (root/'reference.csv').open() as f: rows=list(csv.DictReader(f))
    expected={r['id']:int(r['answer']) for r in rows}; ids=list(expected)
    # Start with a previously reliable short problem; each planned attempt is still run.
    order=sorted(ids,key=lambda p:(p!='26de63',ids.index(p)))
    manifest={'repeats':REPEATS,'ids':ids,'source_sha256':{p:file_hash(ROOT/p) for p in ['run_math_harness.py','run_harness_campaign.py','harness/short_answer.txt','start.sh','diagnostics/run_direct_supervised.py']}}
    if (root/'manifest.json').exists() and json.loads((root/'manifest.json').read_text())!=manifest:
        raise RuntimeError('campaign source/config mismatch; inspect before resuming')
    write_json(root/'manifest.json',manifest)
    (root/'logs').mkdir(exist_ok=True)
    previous=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    if previous.get('child_pid'):
        stop_owned(previous['child_pid'],previous.get('child_start_ticks'),'run_math_harness.py')
    if previous.get('server'):
        old=previous['server']
        stop_owned(old['pid'],old['start_ticks'],'sglang.launch_server')
    status={'state':'waiting_for_direct','pid':os.getpid(),'start_ticks':ticks(os.getpid()),'updated_at':now()}
    service=None; child=None; child_ticks=None; results=[]
    def save():
        status['updated_at']=now();write_json(root/'status.json',status)
    def interrupt(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    save();report(root,ids,results,status['state'],args.direct_run)
    try:
        while True:
            direct=json.loads((args.direct_run/'campaign-status.json').read_text())
            if direct['state']=='completed' and direct.get('server_released'):
                break
            if direct['state'] in {'failed','interrupted'}:raise RuntimeError('direct campaign did not complete')
            time.sleep(args.interval)
        status['state']='starting_server';save()
        servers=root/'servers';servers.mkdir(exist_ok=True)
        serve_dir=servers/f'launch-{len(list(servers.glob("launch-*")))+1:03d}'
        env=os.environ.copy();env.update(RUN_DIR=str(serve_dir),POCHI_MAX_RUNNING_REQUESTS='64',
                                        POCHI_QUANTIZATION='fp8',POCHI_KV_DTYPE='fp8_e4m3',
                                        POCHI_ATTENTION_BACKEND='trtllm_mha',POCHI_CONTEXT_LENGTH='262144',POCHI_DFLASH='0')
        with (root/'logs/start.log').open('a') as log:
            subprocess.run([str(ROOT/'start.sh')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        service=json.loads((serve_dir/'service.json').read_text())
        write_json(root/'service.json',service)
        status['server']=service;save()
        import urllib.request
        deadline=time.monotonic()+1800
        while time.monotonic()<deadline:
            if ticks(service['pid'])!=service['start_ticks']:raise RuntimeError('harness server exited at startup')
            try:
                with urllib.request.urlopen('http://127.0.0.1:30000/get_server_info',timeout=10) as response: info=json.load(response)
                assert info['quantization']=='fp8' and info['kv_cache_dtype']=='fp8_e4m3'
                assert info['context_length']==262144
                write_json(root/'server-info.json',info)
                write_json(serve_dir/'server-info.json',info)
                break
            except (OSError,ValueError):time.sleep(10)
        else:raise TimeoutError('harness server startup deadline exceeded')
        report(root,ids,results,'running',args.direct_run)
        for effort,k in REPEATS.items():
            for replicate in range(k):
                for pid in order:
                    task_root=root/'tasks'/effort/pid/f'replicate-{replicate}'
                    if (task_root/'audit.json').exists():
                        results.append(audit_task(task_root,pid,expected[pid],effort,replicate))
                        continue
                    status.update(state='running',effort=effort,replicate=replicate,problem=pid,completed_attempts=len(results));save()
                    command=[sys.executable,str(ROOT/'run_math_harness.py'),'--input',str(root/'problems.csv'),
                             '--budget',effort,'--seed',str(replicate),'--problems',pid,'--run-dir',str(task_root)]
                    with (root/'logs'/f'{effort}-{pid}-{replicate}.log').open('a') as log:
                        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                        child_ticks=ticks(child.pid)
                        status.update(child_pid=child.pid,child_start_ticks=child_ticks);save()
                        while True:
                            try:code=child.wait(timeout=args.interval);break
                            except subprocess.TimeoutExpired:
                                if ticks(service['pid'])!=service['start_ticks']:raise RuntimeError('owned harness server exited')
                                if shutil.disk_usage(root).free<16*1024**3:raise RuntimeError('disk reserve reached; safely pausing')
                                statefile=task_root/'status.json'
                                if statefile.exists():status['task_status']=json.loads(statefile.read_text())
                                calls=task_root/'problems/row-0000/calls.jsonl'
                                if calls.exists():status['calls_log_bytes']=calls.stat().st_size
                                save()
                        if code not in (0,2):raise RuntimeError(f'harness {effort}/{pid}/{replicate} exited {code}')
                    result=audit_task(task_root,pid,expected[pid],effort,replicate)
                    results.append(result)
                    status['completed_attempts']=len(results);save()
                    report(root,ids,results,'running',args.direct_run)
                    print(json.dumps({k:result[k] for k in ['effort','id','replicate','answer','correct','calls','completion_tokens']}),flush=True)
        status['state']='completed';save()
    except BaseException as error:
        status.update(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error));save()
        raise
    finally:
        if child is not None and child.poll() is None:
            stop_owned(child.pid,child_ticks,'run_math_harness.py');child.wait(timeout=15)
        if service is not None:
            stop_owned(service['pid'],service['start_ticks'],'sglang.launch_server')
            status['server_released']=ticks(service['pid'])!=service['start_ticks']
            (root/'gpu-after.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
        save();report(root,ids,results,status['state'],args.direct_run)
        write_json(root/'PAUSED.json',{'state':status['state'],'server_released':status.get('server_released'), 'updated_at':now()})


if __name__=='__main__':main()
