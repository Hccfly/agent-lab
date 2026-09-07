"""验证评测器能抓失败：答案正确也不能掩盖错工具、缺回填和提前合流。"""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_compare import Recorder, evaluation_environment, protocol_errors, run_case, score, summarize
from eval_scenarios import CASES, Message, ScenarioClient


class CompareTests(unittest.TestCase):
    def test_paired_scenarios(self):
        with evaluation_environment("offline"):
            rows = [run_case(c, engine, "offline") for c in CASES for engine in ("v1", "v2")]
        for row in rows:
            with self.subTest(case=row["case_id"], engine=row["engine"]):
                self.assertTrue(row["passed"], row["reasons"])
                self.assertIsNone(row["tokens"]["total_tokens"])
        for row in [r for r in rows if r["case_id"] == "dedup"]:
            self.assertEqual((row["tool_requests"], row["tool_executions"]), (2, 1))
        aggregate = summarize(rows)
        self.assertEqual(aggregate["v1"]["model_calls"], 23)
        self.assertEqual(aggregate["v2"]["model_calls"], 23)
        self.assertIsNone(aggregate["v1"]["total_tokens"])

    def test_correct_answer_wrong_tool_fails(self):
        case = next(c for c in CASES if c["id"] == "arithmetic")
        recorder = Recorder(None, case, "offline")
        recorder.models = [dict(tool_requests=["calculator"])]
        recorder.tools = [dict(name="get_current_time", arguments={}, start=0, end=1, ok=True)]
        self.assertFalse(score(case, "6731", "completed", recorder)[0])
        recorder.tools = [dict(name="calculator", arguments={"expression": "127*53"}, start=0, end=1, ok=False)]
        self.assertFalse(score(case, "6731", "completed", recorder)[0])

    def test_parallel_order_allowed_but_premature_join_fails(self):
        case = next(c for c in CASES if c["id"] == "multi_diamond")
        recorder = Recorder(None, case, "offline")
        recorder.models = [dict(tool_requests=["calculator"] * 3)]
        recorder.tools = [dict(name="calculator", arguments={"expression": expr}, start=start, end=end, ok=True)
                          for expr, start, end in [("7*8", 0, 3), ("12*5", 1, 2), ("60+56", 4, 5)]]
        self.assertTrue(score(case, "116", "completed", recorder)[0])
        recorder.tools[-1]["start"] = 2.5
        self.assertFalse(score(case, "116", "completed", recorder)[0])

    def test_missing_or_mismatched_tool_response_fails(self):
        valid = [{"role": "assistant", "tool_calls": [{"id": "a"}]},
                 {"role": "tool", "tool_call_id": "a", "content": "3"}]
        self.assertFalse(protocol_errors(valid))
        self.assertTrue(protocol_errors(valid[:1]))
        invalid = deepcopy(valid)
        invalid[1]["tool_call_id"] = "b"
        self.assertTrue(protocol_errors(invalid))

    def test_usage_all_rounds_and_missing_data(self):
        responses = iter([
            SimpleNamespace(choices=[SimpleNamespace(message=Message("answer"))],
                            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)),
            SimpleNamespace(choices=[SimpleNamespace(message=Message("answer"))], usage=None),
        ])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: next(responses))))
        recorder = Recorder(client, CASES[0], "offline")
        recorder.create(messages=[])
        self.assertEqual(recorder.tokens(), dict(prompt_tokens=10, completion_tokens=5, total_tokens=15))
        recorder.create(messages=[])
        self.assertIsNone(recorder.tokens()["total_tokens"])

    def test_offline_context_blocks_network_and_restores_environment(self):
        import os
        import socket
        previous = os.environ.get("OPENAI_API_KEY")
        with evaluation_environment("offline"):
            with socket.socket() as sock, self.assertRaisesRegex(AssertionError, "network forbidden"):
                sock.connect(("127.0.0.1", 1))
        self.assertEqual(os.environ.get("OPENAI_API_KEY"), previous)

    def test_live_adapter_collects_usage_without_real_api_and_restores_v1(self):
        from unittest.mock import patch
        with evaluation_environment("offline"):
            import agent
            case = next(c for c in CASES if c["id"] == "arithmetic")

            class UsageClient(ScenarioClient):
                def create(self, **kwargs):
                    response = super().create(**kwargs)
                    response.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
                    return response

            original_runner = agent.run_tool
            for engine in ("v1", "v2"):
                client = UsageClient(case)
                with patch.object(agent, "client", client):
                    row = run_case(case, engine, "live")
                    self.assertIs(agent.client, client)
                    self.assertIs(agent.run_tool, original_runner)
                self.assertTrue(row["passed"], row["reasons"])
                self.assertEqual(row["tokens"]["total_tokens"], 30)
                self.assertEqual(row["usage_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
