"""真 LangGraph + 假模型/工具，屏障验证实际并发而非速度猜测。"""
import json
import os
from pathlib import Path
import re
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 导入也不依赖项目 .env/API key；测试期间禁止任何 socket 连接及云追踪。
with patch("dotenv.load_dotenv"), patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test"}):
    from langgraph_multiagent import (
        DAG_PLANNER_SYSTEM, SUMMARIZER_SYSTEM, MultiAgentDependencies,
        build_multiagent_graph, invoke_multiagent, merge_task_maps,
    )
    from multiagent import ROLE_REGISTRY
    from tools import IDEMPOTENT_TOOLS


def task(tid, dependencies=None):
    return {"task_id": tid, "agent": "calculator", "description": f"JOB{tid}: 计算1+2",
            "depends_on": dependencies or []}


class FakeModel:
    def __init__(self, tasks, barrier=None, failures=None, looping=False):
        self.plan = json.dumps({"tasks": tasks})
        self.barrier = barrier
        self.failures = set(failures or [])
        self.looping = looping
        self.lock = threading.Lock()
        self.started = []
        self.finished = set()
        self.snapshots = {}
        self.prompts = {}
        self.roles = {}
        self.summary_calls = 0
        self.summary_data = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        messages = kwargs["messages"]
        system = messages[0]["content"]
        if system == DAG_PLANNER_SYSTEM:
            content, calls = self.plan, []
        elif system == SUMMARIZER_SYSTEM:
            with self.lock:
                self.summary_calls += 1
                self.summary_data = messages[-1]["content"]
            content, calls = "汇总完成", []
        else:
            prompt = messages[1]["content"]
            tid = int(re.search(r"JOB(\d+)", prompt)[1])
            role = next(r for r, cfg in ROLE_REGISTRY.items() if cfg["system"] == system)
            with self.lock:
                self.roles.setdefault(tid, []).append(role)
            if tid in self.failures:
                raise RuntimeError(f"JOB{tid} 模型故障")
            if messages[-1]["role"] != "tool" or self.looping:
                with self.lock:
                    self.started.append(tid)
                    self.snapshots[tid] = set(self.finished)
                    self.prompts[tid] = prompt
                if self.barrier and tid in (1, 2):
                    self.barrier.wait(timeout=5)
                # 两个同角色任务故意使用相同参数，验证缓存按子任务隔离。
                calls = [SimpleNamespace(id=f"call-{tid}", function=SimpleNamespace(
                    name="calculator", arguments='{"expression":"1+2"}'))]
                content = ""
            else:
                with self.lock:
                    self.finished.add(tid)
                content, calls = f"result-{tid}:3", []
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=content, tool_calls=calls))], usage=None)


class DAGTests(unittest.TestCase):
    def setUp(self):
        self.network = patch("socket.socket.connect", side_effect=AssertionError("测试禁止网络"))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.tracing = patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)

    def run_graph(self, client, max_tasks=10, max_concurrency=4, graph=None):
        executed = []
        lock = threading.Lock()

        def runner(name, arguments):
            with lock:
                executed.append((name, arguments))
            return "3"

        if graph is None:
            graph = build_multiagent_graph(MultiAgentDependencies(
                model_client=client, model="offline", tool_runner=runner,
                role_registry=ROLE_REGISTRY, idempotent_tools=IDEMPOTENT_TOOLS,
                executor_max_iterations=2))
        state = invoke_multiagent("DAG 测试", max_tasks=max_tasks, graph=graph,
                                  max_concurrency=max_concurrency)
        return state, executed, graph

    def test_parallel_fanout_diamond_join_and_isolation(self):
        # 故意乱序：汇合任务在列表最前，不能按列表位置执行。
        client = FakeModel([task(3, [1, 2]), task(2), task(1)], threading.Barrier(2))
        state, executed, graph = self.run_graph(client, max_concurrency=2)
        self.assertEqual(state["stop_reason"], "completed")
        self.assertEqual(state["task_results"], {1: "result-1:3", 2: "result-2:3", 3: "result-3:3"})
        self.assertEqual(len(executed), 3)
        self.assertEqual(client.snapshots[3], {1, 2})
        self.assertNotIn("result-2", client.prompts[1])
        self.assertNotIn("result-1", client.prompts[2])
        self.assertIn("result-1:3", client.prompts[3])
        self.assertIn("result-2:3", client.prompts[3])
        self.assertEqual(client.summary_calls, 1)
        waves = [e["task_ids"] for e in state["trace"] if "task_ids" in e]
        self.assertEqual(waves, [[2, 1], [3]])
        self.assertTrue(all(e["wave"] == 2 for e in state["trace"] if e.get("task_id") == 3))
        # 同一个编译图再次 invoke：结果、缓存、波次都从空状态开始。
        again, _, _ = self.run_graph(client, graph=graph)
        self.assertEqual(again["task_results"], state["task_results"])
        self.assertEqual(again["wave"], 2)

    def test_max_concurrency_one_and_direct_dependencies_only(self):
        client = FakeModel([task(1), task(2), task(3, [1])])
        state, _, _ = self.run_graph(client, max_concurrency=1)
        self.assertEqual(state["stop_reason"], "completed")
        self.assertEqual(client.started, [1, 2, 3])
        self.assertIn("result-1:3", client.prompts[3])
        self.assertNotIn("result-2:3", client.prompts[3])
        self.assertNotIn("result-1:3", client.prompts[2])

    def test_failure_isolation_and_transitive_blocking(self):
        client = FakeModel([task(4, [3]), task(3, [1]), task(2), task(1)], failures=[1])
        state, _, _ = self.run_graph(client)
        self.assertEqual(state["stop_reason"], "partial_failure")
        self.assertEqual(state["task_status"], {1: "failed", 2: "completed", 3: "blocked", 4: "blocked"})
        self.assertEqual(state["task_results"], {2: "result-2:3"})
        self.assertIn("RuntimeError", state["task_errors"][1])
        self.assertNotIn(3, client.roles)
        self.assertNotIn(4, client.roles)
        self.assertEqual(client.roles[1], ["calculator", "general"])
        self.assertIn("blocked", client.summary_data)
        self.assertIn("JOB1 模型故障", client.summary_data)
        self.assertEqual(client.summary_calls, 1)

    def test_model_iteration_limit_blocks_dependents(self):
        client = FakeModel([task(1), task(2, [1])], looping=True)
        state, _, _ = self.run_graph(client)
        self.assertEqual(state["task_status"], {1: "failed", 2: "blocked"})
        self.assertIn("最大迭代次数", state["task_errors"][1])
        self.assertEqual(client.roles[1], ["calculator", "calculator", "general", "general"])

    def test_invalid_dags_do_not_execute(self):
        invalid = [
            [task(1, [99])], [task(1, [1])], [task(1, [2]), task(2, [1])],
            [task(1), task(1)], [task(True)], [task(1, ["2"]), task(2)],
            [{**task(1), "depends_on": "oops"}], [{**task(1), "description": ""}],
        ]
        for tasks in invalid:
            with self.subTest(tasks=tasks):
                client = FakeModel(tasks)
                state, executed, _ = self.run_graph(client)
                self.assertEqual(state["stop_reason"], "invalid_plan")
                self.assertFalse(executed)
                self.assertFalse(client.roles)
                self.assertEqual(client.summary_calls, 0)

    def test_task_limit_rejects_whole_plan_instead_of_cutting_edges(self):
        client = FakeModel([task(1, [2]), task(2)])
        state, executed, _ = self.run_graph(client, max_tasks=1)
        self.assertEqual(state["stop_reason"], "invalid_plan")
        self.assertFalse(executed)

    def test_long_reverse_chain_stays_within_graph_budget(self):
        client = FakeModel([task(i, [i - 1] if i > 1 else []) for i in range(12, 0, -1)])
        state, _, _ = self.run_graph(client, max_tasks=12)
        self.assertEqual(len(state["task_results"]), 12)
        self.assertEqual(state["wave"], 12)
        self.assertEqual(client.started, list(range(1, 13)))

    def test_reducer_does_not_mutate_or_silently_overwrite(self):
        original = {1: "a"}
        self.assertEqual(merge_task_maps(original, {2: "b"}), {1: "a", 2: "b"})
        self.assertEqual(original, {1: "a"})
        with self.assertRaises(ValueError):
            merge_task_maps(original, {1: "different"})

    def test_invalid_invocation_limits(self):
        for value in [0, -1, True, 1.5]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                invoke_multiagent("test", max_concurrency=value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
