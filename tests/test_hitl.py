"""离线验证敏感工具执行前的 approve/reject/edit。"""

import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

with patch("dotenv.load_dotenv"), patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test"}):
    from hitl import ApprovalThreadError, PersistentApprovalAgent
    from langgraph_agent import (
        AgentDependencies,
        _normalize_approval_decisions,
    )
    from tools import IDEMPOTENT_TOOLS, TOOLS


class FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)


class FakeMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeClient:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.received = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.received.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=next(self.messages))],
            usage=None,
        )


def dependencies(client, executed):
    def runner(name, arguments):
        executed.append((name, arguments))
        return f"已读取: {arguments.get('path', arguments)}"

    return AgentDependencies(
        model_client=client,
        model="offline",
        tool_schemas=TOOLS,
        tool_runner=runner,
        system_prompt="offline system",
        idempotent_tools=IDEMPOTENT_TOOLS,
    )


class HumanInTheLoopTests(unittest.TestCase):
    def setUp(self):
        self.network = patch(
            "socket.socket.connect", side_effect=AssertionError("测试禁止网络")
        )
        self.network.start()
        self.addCleanup(self.network.stop)
        self.tracing = patch.dict(
            os.environ,
            {"LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "false"},
        )
        self.tracing.start()
        self.addCleanup(self.tracing.stop)

    def _start_read(self, database, thread_id, path, executed):
        client = FakeClient([
            FakeMessage(tool_calls=[
                FakeToolCall("read-1", "read_file", f'{{"path":"{path}"}}')
            ])
        ])
        with PersistentApprovalAgent(
            database,
            dependencies(client, executed),
        ) as agent:
            paused = agent.start("读取文件", thread_id)
        self.assertTrue(paused.waiting_for_approval)
        self.assertFalse(paused.completed)
        self.assertEqual(paused.pending_calls[0]["name"], "read_file")
        self.assertEqual(executed, [])
        return paused

    def test_approve_after_restart_executes_once(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "hitl.sqlite3"
            executed = []
            self._start_read(database, "approve-demo", "A.txt", executed)

            # 新连接与新 FakeClient 模拟审批发生在另一个进程生命周期。
            resumed_client = FakeClient([FakeMessage("审批后完成")])
            with PersistentApprovalAgent(
                database,
                dependencies(resumed_client, executed),
            ) as agent:
                result = agent.approve("approve-demo")
                self.assertTrue(result.completed)
                self.assertFalse(result.waiting_for_approval)
                with self.assertRaises(ApprovalThreadError):
                    agent.approve("approve-demo")

            self.assertEqual(executed, [("read_file", {"path": "A.txt"})])
            event = next(e for e in result.state["trace"] if e["node"] == "tools")
            self.assertEqual(event["approval"], "approve")
            self.assertEqual(resumed_client.received[0]["messages"][-1]["role"], "tool")

    def test_reject_returns_observation_without_tool_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "hitl.sqlite3"
            executed = []
            self._start_read(database, "reject-demo", "secret.txt", executed)

            resumed_client = FakeClient([FakeMessage("已按要求取消")])
            with PersistentApprovalAgent(
                database,
                dependencies(resumed_client, executed),
            ) as agent:
                result = agent.reject("reject-demo", "文件包含隐私")

            self.assertTrue(result.completed)
            self.assertEqual(executed, [])
            event = next(e for e in result.state["trace"] if e["node"] == "tools")
            self.assertEqual(event["approval"], "reject")
            self.assertIn("文件包含隐私", event["detail"])
            self.assertIn(
                "用户拒绝执行工具调用",
                resumed_client.received[0]["messages"][-1]["content"],
            )

    def test_edit_executes_only_revised_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "hitl.sqlite3"
            executed = []
            paused = self._start_read(database, "edit-demo", "wrong.txt", executed)
            self.assertIn("wrong.txt", paused.pending_calls[0]["arguments"])

            resumed_client = FakeClient([FakeMessage("修改参数后完成")])
            with PersistentApprovalAgent(
                database,
                dependencies(resumed_client, executed),
            ) as agent:
                result = agent.edit("edit-demo", {"path": "safe.txt"})

            self.assertEqual(executed, [("read_file", {"path": "safe.txt"})])
            event = next(e for e in result.state["trace"] if e["node"] == "tools")
            self.assertEqual(event["approval"], "edit")
            self.assertEqual(event["arguments"], {"path": "safe.txt"})

    def test_non_sensitive_tool_does_not_interrupt(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "hitl.sqlite3"
            executed = []
            client = FakeClient([
                FakeMessage(tool_calls=[
                    FakeToolCall("calc-1", "calculator", '{"expression":"1+2"}')
                ]),
                FakeMessage("3"),
            ])
            with PersistentApprovalAgent(
                database,
                dependencies(client, executed),
            ) as agent:
                result = agent.start("计算", "calculator-demo")

            self.assertTrue(result.completed)
            self.assertFalse(result.waiting_for_approval)
            self.assertEqual(executed, [("calculator", {"expression": "1+2"})])

    def test_batch_decisions_must_be_complete_and_match_call_ids(self):
        calls = [
            {"id": "a", "name": "read_file", "arguments": '{"path":"A"}'},
            {"id": "b", "name": "read_file", "arguments": '{"path":"B"}'},
        ]
        invalid = [
            {"decisions": [{"call_id": "a", "action": "approve"}]},
            {"decisions": [{"call_id": "unknown", "action": "approve"}]},
            {"decisions": [
                {"call_id": "a", "action": "approve"},
                {"call_id": "a", "action": "reject", "reason": "no"},
            ]},
            {"decisions": [
                {"call_id": "a", "action": "edit", "arguments": "not-an-object"},
                {"call_id": "b", "action": "approve"},
            ]},
        ]
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(ValueError):
                _normalize_approval_decisions(response, calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
