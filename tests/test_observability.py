"""可观测层离线测试：节点耗时、状态增量、失败位置与报告导出。"""
import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph_agent import AgentDependencies
from langgraph_multiagent import (
    DAG_PLANNER_SYSTEM,
    SUMMARIZER_SYSTEM,
    MultiAgentDependencies,
)
from observability import (
    RunObserver,
    observe_node,
    run_observed_graph,
    write_trace_report,
)
from tools import IDEMPOTENT_TOOLS, TOOLS


class FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)


class FakeMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeClient:
    def __init__(self, messages=None, error=None):
        self._messages = iter(messages or [])
        self._error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **_kwargs):
        if self._error is not None:
            raise self._error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=next(self._messages))],
            usage=None,
        )


class MultiFakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        if system == DAG_PLANNER_SYSTEM:
            message = FakeMessage(
                '{"tasks":[{"task_id":1,"agent":"calculator",'
                '"description":"计算 1+2","depends_on":[]}]}'
            )
        elif system == SUMMARIZER_SYSTEM:
            message = FakeMessage("多 Agent 汇总完成")
        elif kwargs["messages"][-1]["role"] == "tool":
            message = FakeMessage("calculator 完成：3")
        else:
            message = FakeMessage(
                tool_calls=[
                    FakeToolCall("multi-c1", "calculator", '{"expression":"1+2"}')
                ]
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=None,
        )


def dependencies(client, calls):
    def runner(name, arguments):
        calls.append((name, arguments))
        return "3"

    return AgentDependencies(
        model_client=client,
        model="mock-model",
        tool_schemas=TOOLS,
        tool_runner=runner,
        system_prompt="mock system",
        idempotent_tools=IDEMPOTENT_TOOLS,
    )


# 1. 一次工具闭环能看到完整节点路径、耗时和压缩后的 State delta。
call = FakeToolCall("c1", "calculator", '{"expression":"1+2"}')
calls = []
result, observer = run_observed_graph(
    "计算 1+2",
    dependencies=dependencies(
        FakeClient([FakeMessage(tool_calls=[call]), FakeMessage("答案是 3")]),
        calls,
    ),
)
assert result["answer"] == "答案是 3"
assert calls == [("calculator", {"expression": "1+2"})]
events = observer.events
assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
ends = [event for event in events if event["event"] == "node_end"]
assert [event["node"] for event in ends] == ["model", "tools", "model"]
assert all(event["duration_ms"] >= 0 for event in ends)
assert ends[0]["state_delta"]["pending_tool_calls"] == [
    {"id": "c1", "name": "calculator"}
]
assert events[0]["event"] == "run_start"
assert events[-1]["event"] == "run_end" and events[-1]["status"] == "completed"
print("OK  节点路径、耗时、状态增量和运行边界事件完整")


# 2. 多 Agent 父图、Executor 子图、角色单图共用同一事件协议。
calculator_schema = next(
    schema for schema in TOOLS if schema["function"]["name"] == "calculator"
)
role_registry = {
    "calculator": {"system": "calculator role", "tools": [calculator_schema]},
    "general": {"system": "general role", "tools": [calculator_schema]},
}
multi_result, multi_observer = run_observed_graph(
    "完成计算并汇总",
    mode="multi",
    dependencies=MultiAgentDependencies(
        model_client=MultiFakeClient(),
        model="mock-model",
        tool_runner=lambda _name, _arguments: "3",
        role_registry=role_registry,
        idempotent_tools=IDEMPOTENT_TOOLS,
        executor_max_iterations=3,
    ),
)
assert multi_result["answer"] == "多 Agent 汇总完成"
multi_nodes = {
    event["node"]
    for event in multi_observer.events
    if event["event"] == "node_end"
}
assert {
    "planner",
    "scheduler",
    "executor",
    "executor.prepare",
    "executor.run_role",
    "role.calculator.model",
    "role.calculator.tools",
    "summarizer",
} <= multi_nodes
print("OK  多 Agent 父图、子图和角色图可在同一 run_id 下追踪")


# 3. 模型异常保留精确失败节点，并由 run_end 标记整次运行失败。
failure_observer = RunObserver()
try:
    run_observed_graph(
        "触发故障",
        observer=failure_observer,
        dependencies=dependencies(FakeClient(error=RuntimeError("模型暂不可用")), []),
    )
    raise AssertionError("预期模型异常")
except RuntimeError as exc:
    assert "模型暂不可用" in str(exc)

failure_events = failure_observer.events
node_error = next(event for event in failure_events if event["event"] == "node_error")
assert node_error["node"] == "model"
assert node_error["error"] == {"type": "RuntimeError", "message": "模型暂不可用"}
assert failure_events[-1]["event"] == "run_end"
assert failure_events[-1]["status"] == "failed"
print("OK  异常轨迹定位到 model 节点并结构化收尾")


# 4. 并行分支共享观察器时 seq 仍唯一且连续。
parallel_observer = RunObserver()


def echo_node(state):
    return {"task_status": {state["task_id"]: "completed"}}


wrapped = observe_node(parallel_observer, "executor", echo_node)
with ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(wrapped, ({"task_id": item} for item in range(20))))

parallel_events = parallel_observer.events
assert len(parallel_events) == 40
assert [event["seq"] for event in parallel_events] == list(range(1, 41))
assert len({event["span_id"] for event in parallel_events}) == 20
print("OK  Send 风格并行分支下事件序号与 span 配对线程安全")


# 5. 同一轨迹可原子导出 JSON 和带 Mermaid/时间表的 Markdown。
with tempfile.TemporaryDirectory() as temp_dir:
    json_path, markdown_path = write_trace_report(
        Path(temp_dir) / "trace",
        observer,
        metadata={"mode": "offline"},
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["run_id"] == observer.run_id
    assert payload["metadata"]["mode"] == "offline"
    assert "```mermaid" in markdown
    assert "| Seq | 节点 |" in markdown
    assert "`tools`" in markdown
print("OK  JSON/Markdown 双格式轨迹报告可复查、可展示")


print("\n可观测层测试全部通过")
