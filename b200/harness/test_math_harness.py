"""CPU integration checks against the real, unmodified upstream search engine."""
import argparse
import asyncio
import copy
import csv
import fcntl
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_math_harness as runner


class ScriptedClient:
    def __init__(self, *, failure=None, missing_answer=False):
        self.failure = failure
        self.missing_answer = missing_answer
        self.calls = []
        self.messages = {}
        self.closed = False

    async def chat_stream(self, messages, **kwargs):
        return await self.chat_raw(messages, **kwargs)

    async def chat_raw(self, messages, *, request_id, **kwargs):
        self.calls.append(request_id)
        self.messages[request_id] = messages
        if request_id == self.failure:
            raise RuntimeError("injected transport interruption")
        if "/verify/" in request_id:
            score = 1 if "round-02" in request_id else 0.5
            content = (
                "<evaluation>Checked the derivation and its integer answer.</evaluation>"
                "<suggestions>Justify the equality.</suggestions>"
                f"<score>{score}</score>"
            )
        elif "/select/" in request_id:
            candidates = re.findall(
                r'<candidate id="([^"]+)">(.*?)</candidate>',
                messages[-1]["content"], re.S,
            )
            selected = next((cid for cid, proof in candidates if r"\boxed{222}" in proof), candidates[0][0])
            content = f"<selected_id>{selected}</selected_id>"
        else:
            value = 222 if "round-02" in request_id and request_id.endswith("p0001") else 111
            proof = f"A derivation for {request_id}.\n\\boxed{{{value}}}"
            if self.missing_answer:
                proof += "\nThis was an intermediate result; the answer is unknown."
            content = (
                f"<solution>{proof}</solution>"
                "<self_evaluation>All steps checked.</self_evaluation><score>1</score>"
            )
        return {
            "message": {"content": content, "reasoning_content": "retained reasoning"},
            "finish_reason": "stop", "prompt_tokens": 10, "completion_tokens": 20,
            "reasoning_tokens": 5, "cached_prompt_tokens": 0,
            "requested_max_completion_tokens": kwargs["max_completion_tokens"],
            "logical_max_completion_tokens": kwargs["max_completion_tokens"],
            "physical_request_count": 1, "physical_prompt_tokens": 10,
            "segments": [{"kind": "chat", "finish_reason": "stop"}], "latency_s": 0.01,
        }

    async def aclose(self):
        self.closed = True


class MathHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "questions.csv"
        self.input.write_text('id,problem\np1,"Find the integer."\n')
        self.config = self.root / "config.yaml"
        config = runner.load_config(ROOT.parent / "config-model-step225-budget-medium.yaml")
        config["search"].update(
            proofs_per_round=2, verifications_per_proof=2, top_proofs=2,
            refine_parents=2, reviews_per_refine_parent=1, max_rounds=2,
            min_valid_verifications=1, concurrency=4, max_completion_tokens=128,
            solution_continuation_tokens=32, verifier_continuation_tokens=32,
            selection_max_tokens=64, selection_continuation_tokens=16,
            early_stop_threshold=1.0, selection_votes=3, selection_candidates=2,
            selection_tournament=False, filter_degenerate=False,
        )
        self.config.write_text(yaml.safe_dump(config))
        self.tokenizer = self.root / "tokenizer"
        self.tokenizer.mkdir()
        (self.tokenizer / "tokenizer.json").write_text('{}')
        self.args = argparse.Namespace(
            input=self.input, run_dir=self.root / "run", config=self.config,
            budget="medium", url="http://127.0.0.1:30000/v1", model="fm-pochi",
            tokenizer=self.tokenizer, problems="all", limit=0, prepare_only=False,
        )

    def execute(self, client, args=None):
        with patch.object(runner, "LocalChatClient", return_value=client):
            return asyncio.run(runner.run(args or self.args))

    def test_original_refinement_and_selector_determine_answer(self):
        client = ScriptedClient()
        self.assertEqual(self.execute(client), 0)
        self.assertTrue(client.closed)
        with (self.args.run_dir / "submission.csv").open() as fh:
            self.assertEqual(list(csv.DictReader(fh)), [{"id": "p1", "answer": "222"}])
        final = json.loads((self.args.run_dir / "problems/row-0000/final.json").read_text())
        self.assertEqual(final["rounds_completed"], 2)
        self.assertEqual(final["selected_proof_id"], "r02-p0001")
        self.assertIn("llm_selector", final["final_source"])
        self.assertEqual(len(client.calls), 15)  # 4 generations + 8 reviews + 3 ballots
        refinements = [m[-1]["content"] for rid, m in client.messages.items() if rid.startswith("round-02/generate/")]
        self.assertEqual(len(refinements), 2)
        self.assertTrue(all("<verifier_review " in text for text in refinements))
        self.assertTrue(all(runner.CONTRACT.read_text().strip() in m[-1]["content"] for m in client.messages.values()))
        calls = [json.loads(line) for line in (self.args.run_dir / "problems/row-0000/calls.jsonl").read_text().splitlines()]
        self.assertTrue(all(c["reasoning_content"] == "retained reasoning" for c in calls))
        # Three candidates said 111; the original selector still chose 222.
        replay = ScriptedClient()
        self.assertEqual(self.execute(replay), 0)
        self.assertEqual(replay.calls, [])

    def test_interrupted_round_resumes_missing_calls(self):
        failure = "round-01/verify/r01-p0001/v001"
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.execute(ScriptedClient(failure=failure))
        status = json.loads((self.args.run_dir / "status.json").read_text())
        self.assertEqual(status["state"], "failed")
        replay = ScriptedClient()
        self.assertEqual(self.execute(replay), 0)
        self.assertEqual([c for c in replay.calls if c.startswith("round-01/")], [failure])
        calls = [json.loads(line) for line in (self.args.run_dir / "problems/row-0000/calls.jsonl").read_text().splitlines()]
        successful = [c["sample_id"] for c in calls if c["error"] is None]
        self.assertEqual(len(successful), len(set(successful)))

    def test_missing_final_answer_preserves_proof_and_fails_explicitly(self):
        self.assertEqual(self.execute(ScriptedClient(missing_answer=True)), 2)
        status = json.loads((self.args.run_dir / "status.json").read_text())
        self.assertEqual(status["state"], "completed_with_invalid_answers")
        self.assertEqual(status["invalid_answers"], 1)
        answer = json.loads((self.args.run_dir / "problems/row-0000/answer.json").read_text())
        self.assertIsNone(answer["answer"])
        self.assertTrue((self.args.run_dir / "problems/row-0000/final.json").exists())
        with (self.args.run_dir / "submission.csv").open() as fh:
            self.assertEqual(list(csv.DictReader(fh)), [{"id": "p1", "answer": ""}])

    def test_prepare_is_offline_and_original_budget_is_unchanged(self):
        args = copy.copy(self.args)
        args.prepare_only = True
        args.config = None
        with patch.object(runner, "LocalChatClient") as client:
            self.assertEqual(asyncio.run(runner.run(args)), 0)
            client.assert_not_called()
        manifest = json.loads((args.run_dir / "manifest.json").read_text())
        original = runner.load_config(ROOT.parent / "config-model-step225-budget-medium.yaml")
        self.assertEqual(manifest["search"], original["search"])
        for name in ["prover.txt", "verifier.txt", "refiner.txt", "selector.txt"]:
            self.assertIn(f"evaluation/prompts/ycchen_math_3r/{name}", manifest["source_sha256"])

    def test_changed_input_cannot_reuse_cached_answers(self):
        self.execute(ScriptedClient())
        self.input.write_text('id,problem\np1,"A different question."\n')
        with self.assertRaisesRegex(ValueError, "resume mismatch"):
            self.execute(ScriptedClient())

    def test_changed_search_cannot_reuse_cached_answers(self):
        self.execute(ScriptedClient())
        config = yaml.safe_load(self.config.read_text())
        config["search"]["seed"] += 1
        self.config.write_text(yaml.safe_dump(config))
        with self.assertRaisesRegex(ValueError, "resume mismatch"):
            self.execute(ScriptedClient())

    def test_same_directory_cannot_run_twice(self):
        self.args.run_dir.mkdir()
        with (self.args.run_dir / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "another harness"):
                self.execute(ScriptedClient())

    def test_final_integer_contract(self):
        valid = {
            r"\boxed{0}": 0, "Derivation.\n\\boxed{-12}\n": -12, r"\boxed{+0007}": 7,
            "Derivation.\n\\[\n\\boxed{32951}\n\\]": 32951,
            r"\(\boxed{391}\).": 391, r"$$\boxed{42}$$": 42,
            r"\begin{equation}\boxed{50}\end{equation}": 50,
        }
        for text, expected in valid.items():
            with self.subTest(text=text):
                self.assertEqual(runner.extract_answer(text), expected)
        for text in [r"\boxed{5} then \boxed{x}", r"\boxed{1/2}", r"\boxed{3.0}", r"\boxed{3", "42", r"\boxed{8} is only intermediate"]:
            with self.subTest(text=text):
                self.assertIsNone(runner.extract_answer(text))

    def test_replicate_seed_is_pinned(self):
        self.args.seed = 3
        self.args.prepare_only = True
        self.execute(ScriptedClient())
        manifest = json.loads((self.args.run_dir / "manifest.json").read_text())
        self.assertEqual(manifest["search"]["seed"], 3)
        self.args.seed = 2
        with self.assertRaisesRegex(ValueError, "resume mismatch"):
            self.execute(ScriptedClient())

    def test_wire_request_ids_are_isolated_across_runs(self):
        seen = []
        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            })
        async def check():
            client = runner.LocalChatClient('http://example/v1', 'fm-pochi', tokenizer_path=self.tokenizer)
            await client._client.aclose()
            client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            for namespace in ['run-a/row-0000', 'run-b/row-0000']:
                client.request_namespace = namespace
                await client.chat_raw([{"role": "user", "content": "question"}],
                                      max_completion_tokens=128, temperature=1.0, top_p=.95,
                                      seed=7, request_id='round-01/generate/r01-p0000')
            await client.aclose()
        asyncio.run(check())
        self.assertNotEqual(seen[0]['rid'], seen[1]['rid'])
        self.assertEqual(seen[0]['messages'], seen[1]['messages'])
        self.assertEqual(seen[0]['seed'], seen[1]['seed'])

    def test_campaign_audit_checks_selected_proof(self):
        from run_harness_campaign import audit_task
        self.execute(ScriptedClient())
        result = audit_task(self.args.run_dir, 'p1', 222, 'medium', 0)
        self.assertTrue(result['correct'])
        self.assertEqual(result['calls'], 15)
        path = self.args.run_dir / 'problems/row-0000/final.json'
        final = json.loads(path.read_text())
        final['final_proof'] = r'\boxed{999}'
        path.write_text(json.dumps(final))
        with self.assertRaises(AssertionError):
            audit_task(self.args.run_dir, 'p1', 222, 'medium', 0)


if __name__ == "__main__":
    unittest.main()
