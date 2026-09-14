#!/usr/bin/env python3
"""Re-audit completed direct samples without making any inference requests."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_direct import Campaign, summarize, write_json


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',required=True,type=Path)
    ap.add_argument('--collector-source',required=True,type=Path)
    args=ap.parse_args();root=args.run_dir.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    expected_hash=manifest['files_sha256']['run_direct.py']
    assert hashlib.sha256(args.collector_source.read_bytes()).hexdigest()==expected_hash, 'original collector source changed'
    for name in ['problems.csv','reference.csv','prompt.txt']:
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==manifest['files_sha256'][name]
    with (root/'reference.csv').open() as f:rows=list(csv.DictReader(f))
    for row in rows:row['answer']=int(row['answer'])
    configuration=argparse.Namespace(**manifest,run_dir=root,prompt_text=(root/'prompt.txt').read_text().strip())
    campaign=Campaign(configuration,rows)
    records=[json.loads(p.read_text()) for p in (root/'samples').glob('*/result.json')]
    assert {(r['id'],r['sample']) for r in records}=={(r['id'],i) for r in rows for i in range(manifest['k'])}
    assert len(records)==len(rows)*manifest['k']
    assert len({r['rid'] for r in records})==len(records)
    started=time.monotonic()
    for index,record in enumerate(records):
        campaign.verify_record(record)
        if (index+1)%10==0:print(f'audited {index+1}/{len(records)}',flush=True)
    summary=summarize(records,rows,manifest['k'])
    finish=max(datetime.fromisoformat(r['finished_at']) for r in records)
    start=min(datetime.fromisoformat(r['finished_at'])-timedelta(seconds=r['elapsed_s']) for r in records)
    summary.update(
        artifact_audit_passed=True, audited_at=datetime.now(timezone.utc).isoformat(),
        audit_seconds=round(time.monotonic()-started,3),
        generation_wall_seconds=round((finish-start).total_seconds(),3),
        generation_started_at=start.isoformat(), generation_finished_at=finish.isoformat(),
        physical_requests=len(list((root/'samples').glob('*/attempt-*'))),
        failed_attempts=len(list((root/'samples').glob('*/attempt-*/error.json'))),
        sampling={k:manifest[k] for k in ['temperature','top_p','top_k','max_tokens','seed']},
        collector_sha256=expected_hash,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        parser_sha256=hashlib.sha256((ROOT/'run_direct.py').read_bytes()).hexdigest(),
        audit_repair='Linear SSE parsing and chunked replay; original requests, responses and scores are unchanged.',
    )
    summary['aggregate_output_tokens_per_second']=summary['completion_tokens']/summary['generation_wall_seconds']
    write_json(root/'summary.json',summary)
    ordered=sorted(records,key=lambda r:(r['id'],r['sample']))
    (root/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ordered))
    state=json.loads((root/'status.json').read_text())
    state.update(state='completed',updated_at=summary['audited_at'],artifact_audit_passed=True,audit_repaired=True)
    write_json(root/'status.json',state)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
