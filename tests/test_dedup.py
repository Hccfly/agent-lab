"""工具层幂等去重测试:验证同参重复调用只执行一次、非幂等不去重等。

设计要点:幂等知识在工具层(IDEMPOTENT_TOOLS + tool_key),
agent 循环只负责查缓存;去重范围是单次 run_agent 内。
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# 让脚本能从项目根目录导入模块(agent/tools 等)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent
from agent import run_agent
from tools import IDEMPOTENT_TOOLS, tool_key


class FakeToolCall:
    def __init__(self, cid, name, args):
        self.id = cid
        self.function = SimpleNamespace(name=name, arguments=args)


class FakeMessage:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, mode="json", exclude_none=True):
        return {"role": "assistant", "content": self.content}


def make_create(responses):
    """responses: [(content, tool_calls 列表), ...],每个元素对应一轮模型输出。"""
    it = iter(responses)

    def fake_create(**kwargs):
        content, tool_calls = next(it)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=FakeMessage(content, tool_calls))],
            usage=None,
        )

    return fake_create


# --- 白名单声明 ---
assert "calculator" in IDEMPOTENT_TOOLS, "calculator 应是幂等工具"
assert "read_file" in IDEMPOTENT_TOOLS, "read_file 应是幂等工具"
assert "get_current_time" not in IDEMPOTENT_TOOLS, "get_current_time 是非幂等工具(每次结果不同),不得去重"
print("OK  幂等白名单:calculator/read_file 声明为幂等,get_current_time 除外")


# --- 测试 1:同参重复调用只真执行一次,第二次复用缓存 ---
calls = []


def fake_run_tool_1(name, arguments):
    calls.append((name, arguments))
    return f"{name}:{arguments.get('expression', '')}"


tc_dup = [
    FakeToolCall("c1", "calculator", '{"expression": "1+2"}'),
    FakeToolCall("c2", "calculator", '{"expression": "1+2"}'),
]
with patch.object(agent.client.chat.completions, "create", make_create([("", tc_dup), ("答案是 3", None)])), \
     patch.object(agent, "run_tool", fake_run_tool_1):
    out = run_agent("算一下 1+2")
assert len(calls) == 1, f"同参重复调用应只执行一次,实际 {len(calls)} 次: {calls}"
assert out == "答案是 3"
print("OK  同参重复调用只执行一次,第二次复用缓存")


# --- 测试 2:非幂等工具(get_current_time)同参也不去重 ---
calls = []


def fake_run_tool_2(name, arguments):
    calls.append(name)
    return "2026-09-01 12:00:00 Monday"


tc_time = [
    FakeToolCall("c1", "get_current_time", "{}"),
    FakeToolCall("c2", "get_current_time", "{}"),
]
with patch.object(agent.client.chat.completions, "create", make_create([("", tc_time), ("好", None)])), \
     patch.object(agent, "run_tool", fake_run_tool_2):
    run_agent("现在几点")
assert len(calls) == 2, f"非幂等工具不应去重,应执行 2 次,实际 {len(calls)} 次"
print("OK  非幂等工具同参不去重,每次都真执行")


# --- 测试 3:tool_key 指纹——参数顺序无关、内容不同则不同、工具名参与 ---
assert tool_key("read_file", {"path": "a.txt", "mode": "r"}) == tool_key("read_file", {"mode": "r", "path": "a.txt"}), \
    "参数键顺序不同但内容相同,应视为同一调用"
assert tool_key("calculator", {"expression": "1+2"}) != tool_key("calculator", {"expression": "2+3"}), \
    "参数不同应生成不同指纹"
assert tool_key("calculator", {"expression": "1+2"}) != tool_key("read_file", {"expression": "1+2"}), \
    "不同工具即使参数相同也应是不同指纹"
print("OK  指纹:参数顺序无关,内容/工具名不同则不同")


# --- 测试 4:不同参数不去重 ---
calls = []


def fake_run_tool_4(name, arguments):
    calls.append(arguments.get("expression"))
    return str(eval(arguments["expression"]))  # 测试里可放心 eval


tc_diff = [
    FakeToolCall("c1", "calculator", '{"expression": "1+2"}'),
    FakeToolCall("c2", "calculator", '{"expression": "2+3"}'),
]
with patch.object(agent.client.chat.completions, "create", make_create([("", tc_diff), ("5", None)])), \
     patch.object(agent, "run_tool", fake_run_tool_4):
    run_agent("算")
assert len(calls) == 2, f"不同参数应各执行一次,实际 {len(calls)} 次"
print("OK  不同参数不去重,各自真执行")


# --- 测试 5:报错不缓存,相同调用下次仍真执行(鼓励重试/换参) ---
calls = []


def fake_run_tool_5(name, arguments):
    calls.append((name, arguments))
    if len(calls) == 1:
        raise ValueError("临时故障")
    return "ok"


with patch.object(agent.client.chat.completions, "create", make_create([("", tc_dup), ("完成", None)])), \
     patch.object(agent, "run_tool", fake_run_tool_5):
    run_agent("再算 1+2")
assert len(calls) == 2, f"报错不应入缓存,相同调用应再次真执行,实际 {len(calls)} 次"
print("OK  报错不缓存,相同调用下次仍真执行")

print("\n全部通过")
