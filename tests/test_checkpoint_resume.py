"""离线验证 SQLite checkpoint、跨进程恢复和完成任务不重跑。"""

import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

with patch("dotenv.load_dotenv"), patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test"}):
    from checkpointing import CheckpointThreadError, PersistentMultiAgent
    from langgraph_multiagent import (
        DAG_PLANNER_SYSTEM,
        SUMMARIZER_SYSTEM,
        MultiAgentDependencies,
    )
    from multiagent import ROLE_REGISTRY
    from tools import IDEMPOTENT_TOOLS


def task(task_id, depends_on=None):
    return {
        "task_id": task_id,
        "agent": "calculator",
        "description": f"JOB{task_id}: 计算 {task_id}+0",
        "depends_on": depends_on or [],
    }


class RestartFakeModel:
    """每次重建实例就相当于重启后重新创建模型客户端。"""

    def __init__(self):
        self.plan = json.dumps({"tasks": [task(1), task(2), task(3, [1, 2])]})
        self.planner_calls = 0
        self.started = []
        self.summary_calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        messages = kwargs["messages"]
        system = messages[0]["content"]
        if system == DAG_PLANNER_SYSTEM:
            self.planner_calls += 1
            content, calls = self.plan, []
        elif system == SUMMARIZER_SYSTEM:
            self.summary_calls += 1
            content, calls = "持久化任务完成", []
        else:
            task_id = int(re.search(r"JOB(\d+)", messages[1]["content"])[1])
            if messages[-1]["role"] != "tool":
                self.started.append(task_id)
                content = ""
                calls = [SimpleNamespace(
                    id=f"call-{task_id}",
                    function=SimpleNamespace(
                        name="calculator",
                        arguments=json.dumps({"expression": f"{task_id}+0"}),
                    ),
                )]
            else:
                content, calls = f"result-{task_id}", []
        message = SimpleNamespace(content=content, tool_calls=calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def dependencies(client, tool_calls):
    def tool_runner(name, arguments):
        tool_calls.append((name, arguments["expression"]))
        return arguments["expression"].split("+")[0]

    return MultiAgentDependencies(
        model_client=client,
        model="offline",
        tool_runner=tool_runner,
        role_registry=ROLE_REGISTRY,
        idempotent_tools=IDEMPOTENT_TOOLS,
        executor_max_iterations=2,
    )


class CheckpointResumeTests(unittest.TestCase):
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

    def test_restart_resumes_without_repeating_completed_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "checkpoint.sqlite3"
            tool_calls = []

            first_client = RestartFakeModel()
            with PersistentMultiAgent(
                database,
                dependencies(first_client, tool_calls),
                interrupt_after=["executor"],
            ) as runner:
                first = runner.start("完成三个任务", "resume-demo", max_tasks=3)
                self.assertFalse(first.completed)
                self.assertEqual(first.state["task_status"], {1: "completed", 2: "completed"})
                self.assertEqual(first_client.planner_calls, 1)
                self.assertCountEqual(first_client.started, [1, 2])

            # 新连接 + 新依赖实例模拟进程重启；只执行尚未完成的 task 3。
            second_client = RestartFakeModel()
            with PersistentMultiAgent(
                database,
                dependencies(second_client, tool_calls),
                interrupt_after=["executor"],
            ) as runner:
                second = runner.resume("resume-demo")
                self.assertFalse(second.completed)
                self.assertEqual(second_client.planner_calls, 0)
                self.assertEqual(second_client.started, [3])
                self.assertEqual(second.state["task_status"], {
                    1: "completed", 2: "completed", 3: "completed",
                })

            # 再次重启后只执行 scheduler/summarizer，不重跑任何 Executor 或工具。
            third_client = RestartFakeModel()
            with PersistentMultiAgent(
                database,
                dependencies(third_client, tool_calls),
                interrupt_after=["executor"],
            ) as runner:
                final = runner.resume("resume-demo")
                self.assertTrue(final.completed)
                self.assertEqual(final.state["stop_reason"], "completed")
                self.assertEqual(third_client.started, [])
                self.assertEqual(third_client.summary_calls, 1)

                before = list(tool_calls)
                same = runner.resume("resume-demo")
                self.assertTrue(same.completed)
                self.assertEqual(tool_calls, before)
                with self.assertRaises(CheckpointThreadError):
                    runner.start("不应覆盖", "resume-demo")

            self.assertEqual(len(tool_calls), 3)
            self.assertTrue(database.exists())

    def test_unknown_and_invalid_thread_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            client = RestartFakeModel()
            with PersistentMultiAgent(
                Path(directory) / "checkpoint.sqlite3",
                dependencies(client, []),
            ) as runner:
                with self.assertRaises(CheckpointThreadError):
                    runner.resume("missing")
                for thread_id in ["", "has space", "../escape"]:
                    with self.subTest(thread_id=thread_id), self.assertRaises(
                        CheckpointThreadError
                    ):
                        runner.status(thread_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
