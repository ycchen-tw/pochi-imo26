#!/usr/bin/env python3
"""Run the original FM-Pochi proof search and export its selected integer answer."""
from __future__ import annotations

import argparse
import asyncio
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT.parent
HARNESS = UPSTREAM / "evaluation" / "harness"
CONTRACT = ROOT / "harness" / "short_answer.txt"
DEFAULT_TOKENIZER = Path(
    "/nfs/aimo/shared/fm-pochi/models/opd-32b-bf16-step-225"
)
sys.path.insert(0, str(HARNESS))

from async_client import AsyncChatClient  # noqa: E402
from eval_config import load_config  # noqa: E402
from proof_search import ProblemSearch, atomic_json  # noqa: E402
from run_submission import InputRow, load_test_csv, select_problems  # noqa: E402


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_answer(proof: str) -> int | None:
    """Only the selected solution's final box; never backscan to an earlier one."""
    boxes = list(re.finditer(r"\\boxed\s*\{", proof))
    if not boxes:
        return None
    tail = proof[boxes[-1].start():]
    match = re.match(r"\\boxed\s*\{\s*([+-]?[0-9]+)\s*\}", tail)
    if match is None:
        return None
    suffix = re.sub(r"\s+", "", tail[match.end():])
    if suffix.endswith((".", "。")):
        suffix = suffix[:-1]
    closers = {"", r"\]", r"\)", "$", "$$"}
    closers.update(r"\end{" + name + "}" for name in
                   ["equation", "equation*", "align", "align*", "gather", "gather*"])
    return int(match[1]) if suffix in closers else None


class LocalChatClient(AsyncChatClient):
    """Use the API's model alias while retaining the local continuation tokenizer."""

    def __init__(self, base_url: str, model: str, *, tokenizer_path: Path, **kwargs):
        super().__init__(base_url, model, **kwargs)
        self.tokenizer_path = tokenizer_path
        self.request_namespace = ""

    def _request_id(self, value):
        prefix = self.request_namespace + "/" if self.request_namespace else ""
        return value if value.startswith(prefix) else prefix + value

    async def chat_raw(self, messages, *, request_id, **kwargs):
        return await super().chat_raw(messages, request_id=self._request_id(request_id), **kwargs)

    async def chat_stream(self, messages, *, request_id, **kwargs):
        return await super().chat_stream(messages, request_id=self._request_id(request_id), **kwargs)

    async def _continue_xml_raw(self, initial, messages, *, request_id, **kwargs):
        return await super()._continue_xml_raw(initial, messages, request_id=self._request_id(request_id), **kwargs)

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.tokenizer_path), local_files_only=True
            )
        return self._tokenizer


def csv_bytes(columns: list[str], rows: list[dict]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def atomic_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def prepare(args: argparse.Namespace) -> tuple[list[InputRow], dict]:
    """Pin all solver inputs before allowing original per-call checkpoints to resume."""
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    config_path = args.config or UPSTREAM / f"config-model-step225-budget-{args.budget}.yaml"
    config = load_config(config_path)
    if getattr(args, "seed", None) is not None:
        if args.seed < 0:
            raise ValueError("--seed must be >= 0")
        config["search"]["seed"] = args.seed
    rows = select_problems(load_test_csv(args.input), args.problems, args.limit)
    contract = CONTRACT.read_text().strip()
    if not contract:
        raise ValueError("short-answer contract is empty")
    tokenizer = args.tokenizer.resolve()
    if not (tokenizer / "tokenizer.json").is_file():
        raise ValueError(f"local tokenizer missing: {tokenizer}")
    adapted = [InputRow(r.id, r.problem + "\n\n" + contract) for r in rows]
    source_paths = sorted(HARNESS.glob("*.py")) + sorted(
        (UPSTREAM / "evaluation" / "prompts" / "ycchen_math_3r").glob("*.txt")
    )
    source_paths += [Path(__file__).resolve(), CONTRACT]
    tokenizer_paths = sorted(tokenizer.glob("*.json")) + sorted(tokenizer.glob("*.jinja"))
    manifest = {
        "schema_version": 1,
        "engine": "upstream.evaluation.harness.proof_search.ProblemSearch",
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {str(p.relative_to(UPSTREAM)): sha256(p.read_bytes()) for p in source_paths},
        "input_sha256": sha256(args.input.read_bytes()),
        "config_sha256": sha256(config_path.read_bytes()),
        "selected_ids": [r.id for r in rows],
        "contract": contract,
        "search": config["search"],
        "base_url": args.url.rstrip("/"),
        "served_model": args.model,
        "tokenizer_path": str(tokenizer),
        "tokenizer_sha256": {p.name: sha256(p.read_bytes()) for p in tokenizer_paths},
    }
    pins = {
        "input.csv": args.input.read_bytes(),
        "config.yaml": config_path.read_bytes(),
        "adapted.csv": csv_bytes(["id", "problem"], [{"id": r.id, "problem": r.problem} for r in adapted]),
    }
    manifest_path = args.run_dir / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("resume mismatch: input, search config, prompt, source or endpoint changed; use a new --run-dir")
    elif any(p.name not in {".lock", *pins} for p in args.run_dir.iterdir()):
        raise ValueError("run directory has unpinned artifacts; use an empty --run-dir")
    for name, data in pins.items():
        path = args.run_dir / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"resume mismatch in pinned file: {name}")
    for name, data in pins.items():
        if not (args.run_dir / name).exists():
            atomic_bytes(args.run_dir / name, data)
    if not manifest_path.exists():
        atomic_json(manifest_path, manifest)
    return adapted, config["search"]


async def solve_rows(args: argparse.Namespace, rows: list[InputRow], search_config: dict) -> int:
    client = LocalChatClient(
        args.url.rstrip("/"), args.model, tokenizer_path=args.tokenizer.resolve(),
        api_key="EMPTY", max_connections=search_config["concurrency"] + 8,
        timeout=float(search_config["request_timeout_seconds"]),
    )
    semaphore = asyncio.Semaphore(search_config["concurrency"])
    answers: list[dict] = []
    status = {"state": "running", "started_at": timestamp(), "total": len(rows), "completed": 0, "invalid_answers": 0}

    def save_status():
        atomic_json(args.run_dir / "status.json", {**status, "updated_at": timestamp()})

    save_status()
    try:
        for index, row in enumerate(rows):
            internal_id = f"row-{index:04d}"
            client.request_namespace = sha256(str(args.run_dir).encode())[:12] + "/" + internal_id
            problem_dir = args.run_dir / "problems" / internal_id
            status.update(current_id=row.id, current_round=None)
            save_status()

            async def checkpoint(value: dict) -> None:
                # Round winners remain provisional until the original selector finishes.
                atomic_json(problem_dir / "answer-checkpoint.json", {
                    **value, "id": row.id, "provisional_answer": extract_answer(value["proof"]),
                })
                status["current_round"] = value["round"]
                save_status()
                print(f"id={row.id} round={value['round']} selected={value['selected_proof_id']}", flush=True)

            search = ProblemSearch(
                problem_id=internal_id, problem=row.problem, output_dir=problem_dir,
                client=client, semaphore=semaphore, config=search_config,
                on_round_complete=checkpoint,
            )
            result = await search.solve()
            answer = extract_answer(result["final_proof"])
            atomic_json(problem_dir / "answer.json", {
                "id": row.id, "answer": answer,
                "state": "answered" if answer is not None else "invalid_answer",
                "selected_proof_id": result["selected_proof_id"],
                "final_source": result["final_source"],
                "proof_artifact": "final.json",
            })
            answers.append({"id": row.id, "answer": "" if answer is None else answer})
            atomic_bytes(args.run_dir / "submission.csv", csv_bytes(["id", "answer"], answers))
            status["completed"] = len(answers)
            status["invalid_answers"] += int(answer is None)
            save_status()
            print(f"id={row.id} answer={answer} selected={result['selected_proof_id']}", flush=True)
        status["state"] = "completed" if not status["invalid_answers"] else "completed_with_invalid_answers"
        status["finished_at"] = timestamp()
        save_status()
        return 2 if status["invalid_answers"] else 0
    except BaseException as error:
        status.update(state="interrupted" if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt)) else "failed", error=repr(error))
        save_status()
        raise
    finally:
        await client.aclose()


async def run(args: argparse.Namespace) -> int:
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another harness owns this --run-dir") from None
        rows, search_config = prepare(args)
        if args.prepare_only:
            if not (args.run_dir / "status.json").exists():
                atomic_json(args.run_dir / "status.json", {"state": "prepared", "total": len(rows), "updated_at": timestamp()})
            print(json.dumps({"run_dir": str(args.run_dir), "problems": len(rows), "search": search_config}, indent=2))
            return 0
        return await solve_rows(args, rows, search_config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV with exactly id,problem")
    parser.add_argument("--run-dir", required=True, type=Path, help="Unique directory under $ARC_RUNS; reuse to resume")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--budget", choices=["medium", "high", "xhigh"], default="medium")
    budget.add_argument("--config", type=Path, help="Original schema-12 YAML; only its search section is executed")
    parser.add_argument("--url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="fm-pochi", help="Served API model name")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER, help="Local target tokenizer for native continuations")
    parser.add_argument("--problems", default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, help="Independent harness replicate; default is the original preset seed")
    parser.add_argument("--prepare-only", action="store_true", help="Pin inputs and show settings without contacting the model")
    args = parser.parse_args()
    try:
        async def entry():
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
            return await run(args)
        code = asyncio.run(entry())
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
