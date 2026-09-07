"""多 Agent 图离线测试：Planner/Executor 子图/Summarizer。"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph_multiagent import (
    DAG_PLANNER_SYSTEM as PLANNER_SYSTEM,
    SUMMARIZER_SYSTEM,
    MultiAgentDependencies,
    build_executor_subgraph,
    build_multiagent_graph,
    invoke_multiagent,
)
from multiagent import ROLE_REGISTRY
from tools import IDEMPOTENT_TOOLS


class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)


class FakeMessage:
    def __init__(self, content: str = "", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


def fake_response(message: FakeMessage):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=None,
    )


def schema_names(schemas: list[dict]) -> tuple[str, ...]:
    return tuple(schema["function"]["name"] for schema in schemas)


class RoutedFakeClient:
    """按 system prompt 路由预设回复，同时记录每个角色实际拿到的工具。"""

    def __init__(self, planner_text: str, summary_text: str = "最终汇总完成"):
        self.planner_text = planner_text
        self.summary_text = summary_text
        self.calls = []
        self.role_tool_sets: list[tuple[str, tuple[str, ...]]] = []
        self.executor_prompts: list[tuple[str, str]] = []
        self.summarizer_prompt = ""
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create),
        )

    def _role_for_system(self, system: str) -> str:
        for role, cfg in ROLE_REGISTRY.items():
            if system == cfg["system"]:
                return role
        raise AssertionError(f"无法识别 Executor system prompt: {system[:80]}")

    def create(self, **kwargs):
        self.calls.append(kwargs)
        messages = kwargs["messages"]
        system = messages[0]["content"]
        if system == PLANNER_SYSTEM:
            return fake_response(FakeMessage(self.planner_text))
        if system == SUMMARIZER_SYSTEM:
            self.summarizer_prompt = messages[-1]["content"]
            return fake_response(FakeMessage(self.summary_text))

        role = self._role_for_system(system)
        tools = schema_names(kwargs["tools"])
        self.role_tool_sets.append((role, tools))
        if messages[-1]["role"] == "tool":
            return fake_response(
                FakeMessage(f"{role}完成:{messages[-1]['content']}")
            )

        self.executor_prompts.append((role, messages[-1]["content"]))
        tool_name = tools[0]
        arguments = {
            "calculator": {"expression": "12*5"},
            "get_current_time": {},
            "read_file": {"path": "D:/fake/notes.txt"},
        }[tool_name]
        return fake_response(
            FakeMessage(
                tool_calls=[
                    FakeToolCall(
                        f"{role}-call",
                        tool_name,
                        json.dumps(arguments, ensure_ascii=False),
                    )
                ]
            )
        )


def make_dependencies(client, tool_runner):
    return MultiAgentDependencies(
        model_client=client,
        model="mock-model",
        tool_runner=tool_runner,
        role_registry=ROLE_REGISTRY,
        idempotent_tools=IDEMPOTENT_TOOLS,
        executor_max_iterations=3,
    )


# 1. 父图和 Executor 子图职责边界显式存在。
noop_client = RoutedFakeClient('{"tasks": []}')
deps = make_dependencies(noop_client, lambda name, arguments: "unused")
parent_graph = build_multiagent_graph(deps)
executor_graph = build_executor_subgraph(deps)
assert {"__start__", "planner", "executor", "summarizer", "__end__"} <= set(
    parent_graph.get_graph().nodes
)
assert {"__start__", "prepare", "run_role", "retry_general", "__end__"} <= set(
    executor_graph.get_graph().nodes
)
print("OK  父图 Planner/Executor/Summarizer 与 Executor 子图结构正确")


# 2. 三个专用角色按顺序执行，最后由独立 Summarizer 汇总。
plan = json.dumps(
    {
        "tasks": [
            {
                "task_id": 1,
                "agent": "calculator",
                "description": "计算 12*5",
                "reason": "需要精确计算",
            },
            {
                "task_id": 2,
                "agent": "time",
                "description": "查询当前时间",
                "reason": "需要实时信息",
                "depends_on": [1],
            },
            {
                "task_id": 3,
                "agent": "file",
                "description": "读取笔记",
                "reason": "需要读取文件",
                "depends_on": [1, 2],
            },
        ]
    },
    ensure_ascii=False,
)
client = RoutedFakeClient(plan, "计算、时间和文件任务均已完成。")
real_tool_calls = []


def fake_tool_runner(name, arguments):
    real_tool_calls.append((name, arguments))
    return {
        "calculator": "60",
        "get_current_time": "2026-09-05 10:00:00 Friday",
        "read_file": "项目笔记内容",
    }[name]


graph = build_multiagent_graph(make_dependencies(client, fake_tool_runner))
state = invoke_multiagent("完成三个不同类型的任务", graph=graph)
assert state["answer"] == "计算、时间和文件任务均已完成。"
assert state["stop_reason"] == "completed"
assert set(state["task_results"]) == {1, 2, 3}
assert [name for name, _ in real_tool_calls] == [
    "calculator", "get_current_time", "read_file"
]
assert state["tasks"][2]["depends_on"] == [1, 2]
print("OK  Planner -> 三个 Executor -> Summarizer 完整闭环")


# 3. 每个专用角色只能看到自己的最小工具子集。
observed_tools = {}
for role, tools in client.role_tool_sets:
    observed_tools.setdefault(role, set()).add(tools)
assert observed_tools["calculator"] == {("calculator",)}
assert observed_tools["time"] == {("get_current_time",)}
assert observed_tools["file"] == {("read_file",)}
print("OK  calculator/time/file 均保持最小工具权限")


# 4. 后续任务只通过 Executor 输入拿到已完成结果，Summarizer 收到完整结果表。
prompt_by_role = {role: prompt for role, prompt in client.executor_prompts}
assert "calculator完成:60" in prompt_by_role["time"]
assert "calculator完成:60" in prompt_by_role["file"]
assert "time完成:2026-09-05" in prompt_by_role["file"]
assert "用户原始目标" in client.summarizer_prompt
assert "项目笔记内容" in client.summarizer_prompt
print("OK  父子图上下文适配与 Summarizer 结果注入正确")


# 5. 父图轨迹包含职责节点，子 Agent 内部路径也可检查。
path = [event["node"] for event in state["trace"] if event["node"] != "scheduler"]
assert path == [
    "planner",
    "executor_prepare", "executor_run",
    "executor_prepare", "executor_run",
    "executor_prepare", "executor_run",
    "summarizer",
]
run_events = [event for event in state["trace"] if event["node"] == "executor_run"]
assert all(event["agent_path"] == ["model", "tools", "model"] for event in run_events)
print("OK  父图轨迹与每个基础单 Agent 子路径均可检查")


# 6. Planner 可直接回答时，条件边直接结束，不误调 Executor/Summarizer。
direct_client = RoutedFakeClient("这个问题无需拆解，可以直接回答。")
direct_graph = build_multiagent_graph(
    make_dependencies(direct_client, lambda name, arguments: "不应执行")
)
direct_state = invoke_multiagent("简单问题", graph=direct_graph)
assert direct_state["answer"] == "这个问题无需拆解，可以直接回答。"
assert direct_state["stop_reason"] == "planner_direct"
assert [event["node"] for event in direct_state["trace"]] == ["planner"]
assert len(direct_client.calls) == 1
print("OK  Planner 直接回答路径为 planner -> END")


# 7. 专用 Executor 明确失败时，只用 general 全工具重派一次。
class RetryFakeClient(RoutedFakeClient):
    def create(self, **kwargs):
        messages = kwargs["messages"]
        system = messages[0]["content"]
        if system in {PLANNER_SYSTEM, SUMMARIZER_SYSTEM}:
            return super().create(**kwargs)
        role = self._role_for_system(system)
        tools = schema_names(kwargs["tools"])
        self.calls.append(kwargs)
        self.role_tool_sets.append((role, tools))
        if role == "calculator":
            return fake_response(FakeMessage("无法完成该子任务"))
        return fake_response(FakeMessage("通用角色已恢复任务"))


retry_plan = json.dumps(
    {"tasks": [{"task_id": 1, "agent": "calculator", "description": "复杂任务"}]},
    ensure_ascii=False,
)
retry_client = RetryFakeClient(retry_plan, "恢复后汇总成功")
retry_graph = build_multiagent_graph(
    make_dependencies(retry_client, lambda name, arguments: "不应调用工具")
)
retry_state = invoke_multiagent("测试失败恢复", graph=retry_graph)
retry_events = [
    event for event in retry_state["trace"]
    if event["node"] == "executor_retry_general"
]
assert len(retry_events) == 1
assert retry_events[0]["retried"] is True
assert set(retry_events[0]["tools"]) == {
    "calculator", "get_current_time", "read_file"
}
assert retry_state["task_results"][1] == "通用角色已恢复任务"
print("OK  专用角色失败后仅升级 general 重派一次")

print("\nLangGraph 多 Agent 测试全部通过")
