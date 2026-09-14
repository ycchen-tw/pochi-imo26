#!/usr/bin/env python3
"""Direct, independently seeded samples with raw SSE archives and pass@k audit."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(os.environ.get("FM_POCHI_DATA_ROOT", ROOT / "data"))


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def write_gzip(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb") as output:
        output.write(json.dumps(value, ensure_ascii=False).encode())
    os.replace(temporary, path)


def answer_of(content):
    boxes = list(re.finditer(r"\\boxed\s*\{", content))
    if not boxes:
        return None
    found = re.match(r"\\boxed\s*\{\s*([+-]?[0-9]+)\s*\}", content[boxes[-1].start():])
    return int(found[1]) if found else None


class StreamResult:
    def __init__(self):
        self.buffer = b""
        self.reasoning = []
        self.content = []
        self.finish = None
        self.usage = None
        self.done = False
        self.model = None
        self.response_id = None

    def feed(self, chunk):
        # Split once: repeated split(..., 1) copies the unconsumed suffix for
        # every line, becoming quadratic when replaying a large saved stream.
        lines = (self.buffer + chunk).split(b"\n")
        self.buffer = lines.pop()
        for line in lines:
            self.line(line.rstrip(b"\r"))

    def line(self, line):
        if not line.startswith(b"data:"):
            return
        data = line[5:].strip()
        if data == b"[DONE]":
            self.done = True
            return
        event = json.loads(data)
        if "error" in event:
            raise RuntimeError(f"stream error: {event['error']}")
        self.model = event.get("model", self.model)
        self.response_id = event.get("id", self.response_id)
        if event.get("usage"):
            self.usage = event["usage"]
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                raise RuntimeError("expected one independently sampled completion")
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                self.reasoning.append(delta["reasoning_content"])
            if delta.get("content"):
                self.content.append(delta["content"])
            if choice.get("finish_reason"):
                self.finish = choice["finish_reason"]

    def result(self):
        if self.buffer.strip():
            self.line(self.buffer.rstrip(b"\r"))
            self.buffer = b""
        if not self.done or self.finish not in {"stop", "length"} or not self.usage:
            raise RuntimeError(f"incomplete stream: done={self.done}, finish={self.finish}, usage={self.usage}")
        if type(self.usage.get("completion_tokens")) is not int:
            raise RuntimeError("missing completion token accounting")
        return {
            "id": self.response_id, "model": self.model,
            "choices": [{"index": 0, "finish_reason": self.finish, "message": {
                "role": "assistant", "content": "".join(self.content),
                "reasoning_content": "".join(self.reasoning),
            }}], "usage": self.usage,
        }


def summarize(records, rows, k):
    by_problem = []
    for row in rows:
        items = [r for r in records if r["id"] == row["id"]]
        correct = sum(r["correct"] for r in items)
        valid = [r["answer"] for r in items if r["finish_reason"] == "stop" and r["answer"] is not None]
        counts = Counter(valid)
        highest = max(counts.values(), default=0)
        winners = sorted(a for a, n in counts.items() if n == highest)
        by_problem.append({
            "id": row["id"], "expected": row["answer"], "completed": len(items),
            "correct": correct, "pass_at_k": correct > 0 if len(items) == k else None,
            "answer_counts": dict(counts), "plurality_answer": winners[0] if len(winners) == 1 else None,
            "plurality_tied": len(winners) > 1,
            "length": sum(r["finish_reason"] == "length" for r in items),
            "unparsed": sum(r["answer"] is None for r in items),
            "completion_tokens": sum(r["completion_tokens"] for r in items),
        })
    complete = all(p["completed"] == k for p in by_problem)
    ncorrect = sum(r["correct"] for r in records)
    lengths = sorted(r["completion_tokens"] for r in records)
    percentile = lambda q: lengths[min(len(lengths) - 1, math.ceil(q * len(lengths)) - 1)] if lengths else None
    return {
        "complete": complete, "mode": "direct", "k": k,
        "problems": len(rows), "expected_samples": len(rows) * k, "completed_samples": len(records),
        "correct_samples": ncorrect, "sample_accuracy": ncorrect / len(records) if records else None,
        "solved_problems": sum(p["correct"] > 0 for p in by_problem),
        "pass_at_k": sum(p["correct"] > 0 for p in by_problem) / len(rows) if complete else None,
        "length_samples": sum(p["length"] for p in by_problem),
        "unparsed_samples": sum(p["unparsed"] for p in by_problem),
        "completion_tokens": sum(lengths), "completion_tokens_p50": percentile(.5),
        "completion_tokens_p95": percentile(.95), "completion_tokens_max": max(lengths, default=None),
        "per_problem": by_problem,
    }


class Campaign:
    def __init__(self, args, rows):
        self.args, self.rows, self.root = args, rows, args.run_dir
        self.records = {}
        self.active = {}
        self.errors = []
        self.started = time.monotonic()
        self.state = "running"
        self.semaphore = asyncio.Semaphore(args.concurrency)
        self.request_prefix = digest(str(self.root).encode())[:12]

    def status(self):
        write_json(self.root / "status.json", {
            "state": self.state, "updated_at": now(),
            "completed": len(self.records), "expected": len(self.rows) * self.args.k,
            "correct": sum(r["correct"] for r in self.records.values()),
            "active": self.active, "errors": self.errors,
            "elapsed_this_session_s": round(time.monotonic() - self.started, 1),
        })

    async def heartbeat(self):
        while True:
            self.status()
            await asyncio.sleep(5)

    async def sample(self, client, row, index):
        key = f"{row['id']}#{index}"
        folder = self.root / "samples" / f"{row['id']}-{index:02d}"
        result_path = folder / "result.json"
        if result_path.exists():
            record = json.loads(result_path.read_text())
            self.verify_record(record)
            self.records[key] = record
            return
        folder.mkdir(parents=True, exist_ok=True)
        seed = int.from_bytes(hashlib.sha256(f"{self.args.seed}:{key}".encode()).digest()[:4], "big") % (2**31 - 1)
        async with self.semaphore:
            self.active[key] = {"started_at": now(), "seed": seed}
            self.status()
            for retry in range(self.args.retries + 1):
                attempt = len(list(folder.glob("attempt-*"))) + 1
                attempt_dir = folder / f"attempt-{attempt:02d}"
                attempt_dir.mkdir()
                rid = f"{self.request_prefix}-{row['id']}-{index:02d}-{attempt:02d}"
                body = {
                    "model": self.args.model,
                    "messages": [{"role": "system", "content": self.args.prompt_text}, {"role": "user", "content": row["problem"]}],
                    "temperature": 1.0, "top_p": .95, "top_k": -1,
                    "max_tokens": self.args.max_tokens, "seed": seed, "rid": rid,
                    "stream": True, "stream_options": {"include_usage": True},
                }
                write_gzip(attempt_dir / "request.json.gz", body)
                started = time.monotonic()
                parser = StreamResult()
                try:
                    with gzip.open(attempt_dir / "response.sse.gz", "wb") as raw:
                        async with client.stream("POST", self.args.url.rstrip('/') + '/v1/chat/completions', json=body) as response:
                            response.raise_for_status()
                            async for chunk in response.aiter_bytes():
                                raw.write(chunk)
                                parser.feed(chunk)
                    response = parser.result()
                    write_gzip(attempt_dir / "response.json.gz", response)
                    answer = answer_of(response["choices"][0]["message"]["content"])
                    record = {
                        "id": row["id"], "sample": index, "seed": seed, "rid": rid,
                        "expected": row["answer"], "answer": answer,
                        "correct": parser.finish == "stop" and answer == row["answer"],
                        "finish_reason": parser.finish, "sse_done": parser.done,
                        "completion_tokens": response["usage"]["completion_tokens"],
                        "prompt_tokens": response["usage"].get("prompt_tokens"),
                        "elapsed_s": round(time.monotonic() - started, 3), "finished_at": now(),
                        "artifacts": {name: {"path": str((attempt_dir / name).relative_to(self.root)), "sha256": digest((attempt_dir / name).read_bytes())}
                                      for name in ["request.json.gz", "response.sse.gz", "response.json.gz"]},
                    }
                    write_json(result_path, record)
                    self.records[key] = record
                    print(f"{len(self.records)}/{len(self.rows)*self.args.k} {key} answer={answer} expected={row['answer']} correct={record['correct']} finish={parser.finish} tokens={record['completion_tokens']} elapsed={record['elapsed_s']:.1f}s", flush=True)
                    break
                except BaseException as error:
                    write_json(attempt_dir / "error.json", {"error": repr(error), "rid": rid, "at": now()})
                    try:
                        await client.post(self.args.url.rstrip('/') + '/abort_request', json={"rid": rid}, timeout=10)
                    except Exception:
                        pass
                    if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt)):
                        raise
                    if retry == self.args.retries:
                        self.errors.append({"sample": key, "error": repr(error)})
                        raise
            self.active.pop(key, None)
            self.status()

    def verify_record(self, record):
        for artifact in record["artifacts"].values():
            path = self.root / artifact["path"]
            if digest(path.read_bytes()) != artifact["sha256"]:
                raise RuntimeError(f"artifact SHA256 mismatch: {path}")
            gzip.decompress(path.read_bytes())
        paths = record["artifacts"]
        request = json.loads(gzip.decompress((self.root / paths["request.json.gz"]["path"]).read_bytes()))
        response = json.loads(gzip.decompress((self.root / paths["response.json.gz"]["path"]).read_bytes()))
        parser = StreamResult()
        with gzip.open(self.root / paths["response.sse.gz"]["path"], "rb") as stream:
            while chunk := stream.read(65536):
                parser.feed(chunk)
        if parser.result() != response or request["rid"] != record["rid"] or request["seed"] != record["seed"]:
            raise RuntimeError("request/response/SSE linkage mismatch")
        row = next(r for r in self.rows if r["id"] == record["id"])
        seed = int.from_bytes(hashlib.sha256(f"{self.args.seed}:{record['id']}#{record['sample']}".encode()).digest()[:4], "big") % (2**31 - 1)
        if (not 0 <= record["sample"] < self.args.k or record["seed"] != seed
                or request["messages"] != [{"role": "system", "content": self.args.prompt_text}, {"role": "user", "content": row["problem"]}]
                or request["model"] != self.args.model or request["max_tokens"] != self.args.max_tokens
                or (request["temperature"], request["top_p"], request["top_k"]) != (1.0, .95, -1)):
            raise RuntimeError("request differs from pinned sample inputs")
        answer = answer_of(response["choices"][0]["message"]["content"])
        if (record["answer"] != answer or record["expected"] != row["answer"]
                or record["correct"] != (parser.finish == "stop" and answer == row["answer"])
                or record["completion_tokens"] != response["usage"]["completion_tokens"]
                or record["finish_reason"] != parser.finish):
            raise RuntimeError("result differs from raw response")

    async def run(self):
        heartbeat = asyncio.create_task(self.heartbeat())
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.args.timeout, connect=30),
                                         limits=httpx.Limits(max_connections=self.args.concurrency + 8, max_keepalive_connections=self.args.concurrency + 8)) as client:
                await asyncio.gather(*(self.sample(client, row, index) for index in range(self.args.k) for row in self.rows))
            self.state = "auditing"
            self.status()
            for record in self.records.values():
                self.verify_record(record)
            summary = summarize(list(self.records.values()), self.rows, self.args.k)
            summary["artifact_audit_passed"] = True
            summary["physical_requests"] = len(list((self.root / "samples").glob("*/attempt-*")))
            summary["failed_attempts"] = len(list((self.root / "samples").glob("*/attempt-*/error.json")))
            summary["finished_at"] = now()
            summary["sampling"] = {"temperature": 1.0, "top_p": .95, "top_k": -1, "max_tokens": self.args.max_tokens, "seed_base": self.args.seed}
            summary["elapsed_this_session_s"] = round(time.monotonic() - self.started, 3)
            write_json(self.root / "summary.json", summary)
            ordered = sorted(self.records.values(), key=lambda r: (r["id"], r["sample"]))
            (self.root / "results.jsonl").write_text(''.join(json.dumps(r) + '\n' for r in ordered))
            self.state = "completed"
            print(json.dumps(summary, indent=2), flush=True)
        except BaseException:
            self.state = "interrupted" if asyncio.current_task().cancelling() else "failed"
            raise
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            self.status()


def prepare(args):
    if min(args.k, args.concurrency, args.max_tokens, args.timeout) <= 0 or args.retries < 0:
        raise ValueError("invalid budget")
    with args.problems.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    with args.reference.open(newline="") as fh:
        gold = {r["id"]: int(r["answer"]) for r in csv.DictReader(fh)}
    ids = [r["id"] for r in rows]
    if not rows or len(set(ids)) != len(ids) or set(ids) != set(gold):
        raise ValueError("problem/reference IDs must be nonempty, unique and identical")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", pid) for pid in ids):
        raise ValueError("problem IDs must contain only letters, digits, underscores or hyphens")
    for row in rows:
        row["answer"] = gold[row["id"]]
    args.prompt_text = args.prompt.read_text().strip()
    manifest = {
        "mode": "direct", "k": args.k, "model": args.model, "url": args.url,
        "concurrency": args.concurrency, "max_tokens": args.max_tokens, "seed": args.seed,
        "temperature": 1.0, "top_p": .95, "top_k": -1,
        "files_sha256": {name: digest(path.read_bytes()) for name, path in {
            "problems.csv": args.problems, "reference.csv": args.reference,
            "prompt.txt": args.prompt, "run_direct.py": Path(__file__),
        }.items()},
    }
    path = args.run_dir / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("run manifest mismatch; use a new output directory")
    if not path.exists():
        write_json(path, manifest)
        for name, source in [("problems.csv", args.problems), ("reference.csv", args.reference), ("prompt.txt", args.prompt)]:
            (args.run_dir / name).write_bytes(source.read_bytes())
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problems', type=Path, default=DEFAULT_DATA_ROOT / 'aimo3-reference/problems.csv')
    parser.add_argument('--reference', type=Path, default=DEFAULT_DATA_ROOT / 'aimo3-reference/reference.csv')
    parser.add_argument('--prompt', type=Path, default=ROOT / 'prompt.txt')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--concurrency', type=int, default=100)
    parser.add_argument('--max-tokens', type=int, default=131072)
    parser.add_argument('--timeout', type=int, default=21600)
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', default='fm-pochi')
    parser.add_argument('--url', default='http://127.0.0.1:30000')
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = prepare(args)
        async def entry():
            loop = asyncio.get_running_loop()
            task = asyncio.current_task()
            loop.add_signal_handler(signal.SIGTERM, task.cancel)
            await Campaign(args, rows).run()
        asyncio.run(entry())


if __name__ == '__main__':
    main()
