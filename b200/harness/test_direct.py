import argparse
import asyncio
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_direct as direct


def stream(answer, finish="stop"):
    events = [
        {"id": "reply", "model": "fm-pochi", "choices": [{"index": 0, "delta": {"reasoning_content": "推理"}}]},
        {"id": "reply", "model": "fm-pochi", "choices": [{"index": 0, "delta": {"content": f"\\boxed{{{answer}}}"}, "finish_reason": finish}]},
        {"usage": {"prompt_tokens": 20, "completion_tokens": 50}, "choices": []},
    ]
    return b''.join(b'data: ' + json.dumps(e, ensure_ascii=False).encode() + b'\n\n' for e in events) + b'data: [DONE]\n\n'


class DirectTests(unittest.TestCase):
    def test_stream_preserves_utf8_across_arbitrary_chunks(self):
        data = stream(42)
        for chunk_size in [1, 3, 17, 10000]:
            parser = direct.StreamResult()
            for i in range(0, len(data), chunk_size):
                parser.feed(data[i:i+chunk_size])
            response = parser.result()
            self.assertEqual(response['choices'][0]['message']['reasoning_content'], '推理')
            self.assertEqual(direct.answer_of(response['choices'][0]['message']['content']), 42)

    def test_incomplete_stream_cannot_be_scored(self):
        parser = direct.StreamResult()
        parser.feed(stream(42).replace(b'data: [DONE]\n\n', b''))
        with self.assertRaisesRegex(RuntimeError, 'incomplete stream'):
            parser.result()

    def test_large_archive_replay_matches_chunked_stream(self):
        event = b'data: {"choices":[{"index":0,"delta":{"reasoning_content":"x"}}]}\n\n'
        data = event * 10000 + stream(42)
        whole = direct.StreamResult()
        whole.feed(data)
        chunked = direct.StreamResult()
        for offset in range(0, len(data), 65536):
            chunked.feed(data[offset:offset+65536])
        self.assertEqual(whole.result(), chunked.result())

    def test_passk_counts_questions_and_requires_all_samples(self):
        rows = [{'id': 'a', 'answer': 1}, {'id': 'b', 'answer': 2}]
        records = [
            {'id': 'a', 'correct': True, 'answer': 1, 'finish_reason': 'stop', 'completion_tokens': 10},
            {'id': 'a', 'correct': False, 'answer': 9, 'finish_reason': 'stop', 'completion_tokens': 10},
            {'id': 'b', 'correct': False, 'answer': 2, 'finish_reason': 'length', 'completion_tokens': 10},
            {'id': 'b', 'correct': False, 'answer': None, 'finish_reason': 'stop', 'completion_tokens': 10},
        ]
        result = direct.summarize(records, rows, 2)
        self.assertEqual(result['pass_at_k'], .5)
        self.assertEqual(result['sample_accuracy'], .25)
        self.assertEqual(result['length_samples'], 1)
        self.assertIsNone(direct.summarize(records[:-1], rows, 2)['pass_at_k'])

    def test_real_async_client_archive_audit_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(run_dir=Path(tmp), concurrency=4, k=2, seed=0,
                                      prompt_text='Return an integer.', url='http://test',
                                      model='fm-pochi', max_tokens=128, timeout=30, retries=0)
            rows = [{'id':'a', 'problem':'question a', 'answer':42}, {'id':'b', 'problem':'question b', 'answer':5}]
            requests = []
            def handler(request):
                body = json.loads(request.content)
                requests.append(body)
                self.assertNotIn('answer', body)
                return httpx.Response(200, content=stream(42), headers={'Content-Type':'text/event-stream'})
            original = httpx.AsyncClient
            def factory(**kwargs):
                return original(transport=httpx.MockTransport(handler), **kwargs)
            with patch.object(direct.httpx, 'AsyncClient', side_effect=factory):
                campaign = direct.Campaign(args, rows)
                asyncio.run(campaign.run())
                summary = json.loads((Path(tmp)/'summary.json').read_text())
                self.assertTrue(summary['artifact_audit_passed'])
                self.assertEqual(summary['pass_at_k'], .5)
                self.assertEqual(len(requests), 4)
                self.assertEqual(len({r['seed'] for r in requests}), 4)
                replay = direct.Campaign(args, rows)
                asyncio.run(replay.run())
                self.assertEqual(len(requests), 4)
                record = next(iter(replay.records.values()))
                corrupted = copy.deepcopy(record)
                corrupted['answer'] = 999
                with self.assertRaisesRegex(RuntimeError, 'result differs'):
                    replay.verify_record(corrupted)


if __name__ == '__main__':
    unittest.main()
