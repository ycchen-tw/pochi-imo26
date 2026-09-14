#!/usr/bin/env python3
"""Apply the extraction-only compatibility fix to saved harness answers, without inference."""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_math_harness import extract_answer
from run_direct import write_json


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def outside_parser(path):
    tree=ast.parse(path.read_text())
    tree.body=[node for node in tree.body if not isinstance(node,ast.FunctionDef) or node.name!='extract_answer']
    return ast.dump(tree,include_attributes=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-dir',required=True,type=Path)
    root=ap.parse_args().run_dir.resolve()
    state=json.loads((root/'status.json').read_text())
    for key in ['pid','child_pid']:
        pid=state.get(key)
        if pid and Path(f'/proc/{pid}').exists():
            raise RuntimeError(f'wait until the recorded {key} {pid} exits')
    old=root/'code/run_math_harness.py';new=ROOT/'run_math_harness.py'
    assert outside_parser(old)==outside_parser(new), 'the change must affect only answer extraction'
    old_sha,new_sha=sha(old),sha(new)
    changes=[]
    for path in [root/'manifest.json',*root.glob('tasks/*/*/*/manifest.json')]:
        manifest=json.loads(path.read_text())
        key='source_sha256'
        previous=manifest[key]['run_math_harness.py']
        assert previous in {old_sha,new_sha}
        if previous==old_sha:
            shutil.copy2(path,path.with_name('manifest.before-answer-parser-fix.json'))
            manifest[key]['run_math_harness.py']=new_sha
            write_json(path,manifest)
        if path.parent==root:continue
        task=path.parent;problem=task/'problems/row-0000'
        if not (problem/'final.json').exists():continue
        final=json.loads((problem/'final.json').read_text())
        answer=json.loads((problem/'answer.json').read_text())
        old_answer=answer['answer'];new_answer=extract_answer(final['final_proof'])
        shutil.copy2(problem/'answer.json',problem/'answer.before-parser-fix.json')
        answer.update(answer=new_answer,state='answered' if new_answer is not None else 'invalid_answer')
        write_json(problem/'answer.json',answer)
        with (task/'submission.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['id','answer']);writer.writeheader()
            writer.writerow({'id':answer['id'],'answer':'' if new_answer is None else new_answer})
        status=json.loads((task/'status.json').read_text())
        status.update(state='completed' if new_answer is not None else 'completed_with_invalid_answers',invalid_answers=int(new_answer is None),answer_parser_repaired=True)
        write_json(task/'status.json',status)
        if (task/'audit.json').exists():
            (task/'audit.json').rename(task/'audit.before-answer-parser-fix.json')
        changes.append({'task':str(task),'old_answer':old_answer,'new_answer':new_answer,'final_sha256':sha(problem/'final.json'),'calls_sha256':sha(problem/'calls.jsonl')})
    shutil.copy2(new,root/'code/run_math_harness.after-answer-parser-fix.py')
    write_json(root/'answer-parser-migration.json',{
        'at':datetime.now(timezone.utc).isoformat(),'old_source_sha256':old_sha,'new_source_sha256':new_sha,
        'outside_parser_ast_identical':True,'inference_requests':0,
        'reason':'Accept standard closing LaTeX delimiters after a boxed integer.', 'changes':changes,
    })
    print(json.dumps(changes,indent=2))


if __name__=='__main__':main()
