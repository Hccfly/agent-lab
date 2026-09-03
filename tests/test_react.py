"""ReAct 显式化测试:解析器提取 + 完整四元组循环 + 流式回调 + 幂等去重。

设计要点:ReAct 是纯文本协议,Agent 用解析器从模型自由输出里提取 Action;
对比 agent.py 的原生 tool calling,解析的鲁棒性是核心工程点。
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# 让脚本能从项目根目录导入模块(react/tools 等)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import react
from react import _parse, run_react
from tools import IDEMPOTENT_TOOLS


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeResp:
    def __init__(self, content):
        self.choices = [SimpleNamespace(message=FakeMessage(content))]
        self.usage = None


def make_create(responses):
    """responses: 按顺序返回的模型输出文本列表。"""
    it = iter(responses)

    def fake_create(**kwargs):
        assert kwargs.get("stream") is not True
        assert "tools" not in kwargs, "ReAct 是纯文本协议,不应传 tools schema"
        return FakeResp(next(it))

    return fake_create


# --- 测试 1:解析器——从自由文本里提取 Action 工具名和 JSON 参数 ---
text = """Thought: 我需要知道现在几点。
Action: get_current_time
Action Input: {}
"""
kind, payload = _parse(text)
assert kind == "action", kind
name, raw, args = payload
assert name == "get_current_time" and args == {}
print("OK  解析器:提取 Action 工具名和 JSON 参数")


# --- 测试 2:解析器——容忍 Action 编号、多行 JSON、Final Answer ---
text2 = """Thought: 用户问了一个算式。
Action 1: calculator
Action Input: {
    "expression": "(3 + 5) * 2"
}
"""
kind2, (name2, raw2, args2) = _parse(text2)
assert name2 == "calculator" and args2 == {"expression": "(3 + 5) * 2"}, (name2, raw2, args2)
print("OK  解析器:容忍 Action 编号和跨行 JSON")

kind3, payload3 = _parse("好的,这是最终答案。\nFinal Answer: 结果是 16")
assert kind3 == "final" and payload3 == "结果是 16", (kind3, payload3)
print("OK  解析器:提取 Final Answer")


# --- 测试 3:完整循环——Thought -> Action -> Observation -> Final Answer ---
tool_calls = []


def fake_run_tool(name, arguments):
    tool_calls.append((name, arguments))
    return "2026-09-01 12:00:00 Monday"


with patch.object(react.client.chat.completions, "create",
                  make_create([
                      # 第一轮:模型思考并决定调时间工具
                      "Thought: 需要实时时间。\nAction: get_current_time\nAction Input: {}",
                      # 第二轮:拿到 Observation 后给出最终答案
                      "Final Answer: 现在是 2026-09-01 12:00:00 星期一",
                  ])), \
     patch.object(react, "run_tool", fake_run_tool):
    out = run_react("现在几点", stream=False)
assert tool_calls == [("get_current_time", {})], f"工具应被调用一次: {tool_calls}"
assert "12:00:00" in out, out
print("OK  完整循环:工具被调用,Observation 回填后模型给出 Final Answer")


# --- 测试 4:流式——on_delta 收到逐字片段,完整文本正确累积 ---
got = []
chunk = SimpleNamespace(
    choices=[SimpleNamespace(delta=SimpleNamespace(content="测", tool_calls=None))],
    usage=None,
)


def fake_stream(**kwargs):
    assert kwargs.get("stream") is True
    return iter([chunk, chunk, chunk])


with patch.object(react.client.chat.completions, "create", fake_stream):
    text = react._call_model([], stream=True, on_delta=got.append)
assert text == "测测测" and got == ["测", "测", "测"], (text, got)
print("OK  流式:on_delta 收到逐字片段,完整文本正确累积")


# --- 测试 5:ReAct 模式同样做幂等去重(同 Action 只执行一次) ---
calls = []


def fake_run_tool5(name, arguments):
    calls.append(arguments.get("expression"))
    return str(2 + 3)


react_dedup = [
    # 第一轮:同一 Action 发了两次(模型重复)
    "Thought: 计算。\nAction: calculator\nAction Input: {\"expression\": \"2+3\"}\n"
    "Action: calculator\nAction Input: {\"expression\": \"2+3\"}",
    "Final Answer: 结果是 5",
]
with patch.object(react.client.chat.completions, "create", make_create(react_dedup)), \
     patch.object(react, "run_tool", fake_run_tool5):
    out = run_react("算 2+3", stream=False)
assert "calculator" in IDEMPOTENT_TOOLS
assert len(calls) == 1, f"ReAct 模式下相同 Action 应只执行一次: {calls}"
assert "5" in out
print("OK  ReAct 模式复用幂等去重:相同 Action 只真执行一次")

print("\n全部通过")
