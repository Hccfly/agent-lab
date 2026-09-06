"""LangGraph 单 Agent 离线测试：不请求真实模型 API。"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph_agent import AgentDependencies, build_agent_graph, invoke_agent
from tools import IDEMPOTENT_TOOLS, TOOLS


class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)


class FakeMessage:
    def __init__(self, content: str = "", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeClient:
    def __init__(self, messages):
        self._messages = iter(messages)
        self.received = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.received.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=next(self._messages))], usage=None
        )


def make_graph(messages, runner=None):
    fake_client = FakeClient(messages)
    tool_calls = []

    def default_runner(name, arguments):
        tool_calls.append((name, arguments))
        return "3"

    graph = build_agent_graph(
        AgentDependencies(
            model_client=fake_client,
            model="mock-model",
            tool_schemas=TOOLS,
            tool_runner=runner or default_runner,
            system_prompt="mock system",
            idempotent_tools=IDEMPOTENT_TOOLS,
        )
    )
    return graph, fake_client, tool_calls


# 1. 图拓扑是 model -> tools -> model 的显式循环。
graph, _, _ = make_graph([FakeMessage("unused")])
node_names = set(graph.get_graph().nodes)
assert {"__start__", "model", "tools", "force_stop", "__end__"} <= node_names
print("OK  图拓扑包含 model/tools/force_stop 与 START/END")


# 2. 不需要工具时直接结束，并保留完整可检查状态。
graph, client, _ = make_graph([FakeMessage("你好，我是图 Agent")])
state = invoke_agent("你好", graph=graph)
assert state["answer"] == "你好，我是图 Agent"
assert state["stop_reason"] == "completed"
assert state["iteration"] == 1
assert [event["node"] for event in state["trace"]] == ["model"]
assert client.received[0]["messages"][0]["role"] == "system"
print("OK  无工具调用时 model -> END，显式状态正确")


# 3. 工具调用后结果回填，再由模型生成最终答案。
call = FakeToolCall("c1", "calculator", '{"expression":"1+2"}')
graph, client, calls = make_graph([FakeMessage(tool_calls=[call]), FakeMessage("答案是 3")])
state = invoke_agent("计算 1+2", graph=graph)
assert state["answer"] == "答案是 3"
assert state["iteration"] == 2
assert calls == [("calculator", {"expression": "1+2"})]
assert [message["role"] for message in state["messages"]] == [
    "user", "assistant", "tool", "assistant"
]
assert client.received[1]["messages"][-1]["content"] == "3"
assert [event["node"] for event in state["trace"]] == ["model", "tools", "model"]
print("OK  model -> tools -> model 闭环与消息回填正确")


# 4. 同轮重复的幂等调用只执行一次，轨迹能看见 dedup。
duplicate_calls = [
    FakeToolCall("c1", "calculator", '{"expression":"1+2"}'),
    FakeToolCall("c2", "calculator", '{"expression":"1+2"}'),
]
graph, _, calls = make_graph([FakeMessage(tool_calls=duplicate_calls), FakeMessage("3")])
state = invoke_agent("再算一次", graph=graph)
assert len(calls) == 1
tool_events = [event for event in state["trace"] if event["node"] == "tools"]
assert [event["dedup"] for event in tool_events] == [False, True]
print("OK  幂等工具缓存复用并写入执行轨迹")


# 5. 工具异常作为 Observation 回填；模型仍可恢复并回答。
def failing_runner(name, arguments):
    raise RuntimeError("临时故障")


graph, client, _ = make_graph(
    [FakeMessage(tool_calls=[call]), FakeMessage("工具失败，请稍后重试")],
    runner=failing_runner,
)
state = invoke_agent("计算", graph=graph)
assert state["stop_reason"] == "completed"
assert "工具执行出错: 临时故障" in client.received[1]["messages"][-1]["content"]
assert state["trace"][1]["ok"] is False
print("OK  工具异常回填模型，图可继续运行")


# 6. 达到上限后走 force_stop，不依赖框架抛递归异常。
graph, _, calls = make_graph([FakeMessage(tool_calls=[call])])
state = invoke_agent("一直调用工具", max_iterations=1, graph=graph)
assert state["stop_reason"] == "max_iterations"
assert "最大迭代次数(1)" in state["answer"]
assert calls == [("calculator", {"expression": "1+2"})]
assert state["trace"][-1]["node"] == "force_stop"
print("OK  最大迭代次数由显式路由安全终止")

print("\nLangGraph 单 Agent 测试全部通过")
