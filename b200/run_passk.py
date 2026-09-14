#!/usr/bin/env python3
"""pass@k client for the real 10k-sample job: bounded concurrency, streaming, resumable.

Two sampling modes, so we can measure which one the server actually handles better:
  --mode n     one request per problem with n=k   (k samples share one prefill)
  --mode fanout  k independent requests per problem (radix cache dedupes the prefix)
"""
import argparse, asyncio, csv, json, re, sys, time
from pathlib import Path
import urllib.request

DEFAULT_PROMPT = Path(__file__).with_name("prompt.txt")

def parse_boxed_integer(content):
    boxes = list(re.finditer(r'\\boxed\s*\{', content))
    if not boxes:
        return None
    m = re.match(r'\\boxed\s*\{\s*([+-]?[0-9]+)\s*\}', content[boxes[-1].start():])
    return int(m.group(1)) if m else None

async def post(session_url, body, timeout):
    loop = asyncio.get_running_loop()
    def _do():
        req = urllib.request.Request(session_url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    return await loop.run_in_executor(None, _do)

async def solve(pid, problem, expected, args, sem, out_lock, out_fh, stats):
    url = args.url.rstrip('/') + '/v1/chat/completions'
    body = {"model": "fm-pochi",
            "messages": [{"role": "system", "content": args.system},
                         {"role": "user", "content": problem}],
            "temperature": 1.0, "top_p": 0.95, "top_k": -1,
            "max_tokens": args.max_tokens, "stream": False}
    if args.mode == "n":
        body["n"] = args.k
    async with sem:
        t0 = time.monotonic()
        try:
            raw = await post(url, body, args.timeout)
        except Exception as e:
            rec = {"id": pid, "error": repr(e), "elapsed": time.monotonic() - t0}
            async with out_lock:
                out_fh.write(json.dumps(rec) + "\n"); out_fh.flush()
            stats["errors"] += 1
            return
        elapsed = time.monotonic() - t0
    answers, finish, ntok = [], [], 0
    for ch in raw["choices"]:
        c = (ch["message"].get("content") or "").strip()
        answers.append(parse_boxed_integer(c))
        finish.append(ch["finish_reason"])
    ntok = raw.get("usage", {}).get("completion_tokens", 0)
    hits = None if expected is None else sum(1 for a in answers if a is not None and a == expected)
    rec = {"id": pid, "expected": expected, "n_returned": len(answers),
           "n_correct": hits, "pass_at_k": None if hits is None else hits > 0,
           "answers": answers, "finish_reasons": finish,
           "completion_tokens": ntok, "elapsed": round(elapsed, 2),
           "tps": round(ntok / elapsed, 1) if elapsed else None}
    async with out_lock:
        out_fh.write(json.dumps(rec) + "\n"); out_fh.flush()
    stats["done"] += 1; stats["tokens"] += ntok
    stats["solved"] += int(bool(hits))
    print(f"  {pid}: {len(answers)} samples, {hits} correct, "
          f"{ntok} tok in {elapsed:.0f}s ({ntok/elapsed:.0f} tok/s)"
          if hits is not None else
          f"  {pid}: {len(answers)} samples, unscored, "
          f"{ntok} tok in {elapsed:.0f}s ({ntok/elapsed:.0f} tok/s)", flush=True)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:30000')
    ap.add_argument('--prompt', type=Path, default=DEFAULT_PROMPT,
                    help='File holding the system prompt (default: prompt.txt beside this script)')
    ap.add_argument('--problems', default='/nfs/aimo/shared/fm-pochi/data/aimo3-reference/problems.csv')
    ap.add_argument('--reference', default=None,
                    help='Optional CSV with id,answer; without it samples are saved unscored')
    ap.add_argument('--mode', choices=['n', 'fanout'], default='fanout')
    ap.add_argument('--k', type=int, default=100)
    ap.add_argument('--limit', type=int, default=0, help='only first N problems')
    ap.add_argument('--concurrency', type=int, default=384, help='in-flight requests')
    ap.add_argument('--max-tokens', type=int, default=131072)
    ap.add_argument('--timeout', type=int, default=21600)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    if not args.prompt.is_file():
        raise SystemExit(f'prompt file not found: {args.prompt}')
    args.system = args.prompt.read_text().strip()
    if not args.system:
        raise SystemExit(f'prompt file is empty: {args.prompt}')
    probs = list(csv.DictReader(open(args.problems)))
    refs = {}
    if args.reference:
        refs = {r['id']: int(r['answer']) for r in csv.DictReader(open(args.reference))}
    if args.limit:
        probs = probs[:args.limit]
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if out.exists():  # resume: never redo finished work
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    done_ids.add(r["id"])
        print(f"resuming, {len(done_ids)} already done")
    # Build the unit-of-work list first, then filter. In fanout mode the unit is a
    # single sample ("q1#7"), not the problem, so resume must match at that level.
    units = []
    for p in probs:
        expected = refs.get(p['id'])
        if args.mode == 'n':
            units.append((p['id'], p['problem'], expected))
        else:
            units.extend((f"{p['id']}#{j}", p['problem'], expected) for j in range(args.k))
    todo = [u for u in units if u[0] not in done_ids]
    print(f"{len(todo)} of {len(units)} units to run "
          f"(k={args.k} mode={args.mode} conc={args.concurrency})")
    if not todo:
        print("nothing to do"); return

    sem = asyncio.Semaphore(args.concurrency)
    out_lock = asyncio.Lock()
    stats = {"done": 0, "errors": 0, "tokens": 0, "solved": 0}
    t0 = time.monotonic()
    with out.open("a") as fh:
        await asyncio.gather(*(
            solve(uid, text, expected, args, sem, out_lock, fh, stats)
            for uid, text, expected in todo))
    el = time.monotonic() - t0
    summary = {"mode": args.mode, "k": args.k, "units_run": len(todo), "units_total": len(units),
               "elapsed_seconds": round(el, 1), "tokens": stats["tokens"],
               "aggregate_tps": round(stats["tokens"] / el, 1) if el else None,
               "solved_at_k": stats["solved"], "errors": stats["errors"]}
    print(json.dumps(summary, indent=2))
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")

if __name__ == "__main__":
    asyncio.run(main())
