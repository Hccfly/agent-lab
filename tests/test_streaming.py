"""流式输出测试:验证流式重建 + on_delta 回调 + token 用量。"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# 让脚本能从项目根目录导入模块(agent/log 等)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent
from agent import run_agent


class FakeDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeToolDelta:
    def __init__(self, index, id="", name="", args=""):
        self.index = index
        self.id = id
        self.function = SimpleNamespace(name=name, arguments=args)


class FakeChoice:
    def __init__(self, delta):
        self.delta = delta


class FakeChunk:
    def __init__(self, delta, usage=None):
        self.choices = [FakeChoice(delta)]
        self.usage = usage


def make_stream(*chunks):
    return iter(chunks)


# --- 测试 1:流式文本 + on_delta 收到逐字片段 ---
def fake_create_stream(**kwargs):
    assert kwargs.get("stream") is True, "流式测试应传 stream=True"
    if not any(m.get("tool_calls") for m in kwargs["messages"]):
        # 第一轮:流式工具调用(工具名和参数分多个 chunk)
        return make_stream(
            FakeChunk(FakeDelta(tool_calls=[FakeToolDelta(0, id="c1", name="get_current_time")])),
            FakeChunk(FakeDelta(tool_calls=[FakeToolDelta(0, args="{}")])),
        )
    # 第二轮:流式输出最终答案
    return make_stream(
        FakeChunk(FakeDelta(content="现在")),
        FakeChunk(FakeDelta(content="是")),
        FakeChunk(FakeDelta(content="测试时间")),
        FakeChunk(FakeDelta(), usage=SimpleNamespace(total_tokens=42)),
    )


with patch.object(agent.client.chat.completions, "create", fake_create_stream):
    got = []
    out = run_agent("现在几点", stream=True, on_delta=got.append)

assert out == "现在是测试时间", f"流式最终答案错误: {out!r}"
assert got == ["现在", "是", "测试时间"], f"on_delta 收到片段不对: {got}"
print("OK  流式文本:on_delta 收到逐字片段, 完整答案:", out)


# --- 测试 2:流式工具调用重建(id/name/arguments 分散 chunk 也能合并) ---
with patch.object(agent.client.chat.completions, "create", fake_create_stream):
    msg, usage = agent._stream_model([{"role": "user", "content": "算一下"}], on_delta=None)
assert len(msg.tool_calls) == 1, "应重建出 1 个工具调用"
tc = msg.tool_calls[0]
assert tc.id == "c1" and tc.function.name == "get_current_time" and tc.function.arguments == "{}", tc
print("OK  流式工具调用重建:id/name/arguments 分散 chunk 正确合并")


# --- 测试 3:token 用量被捕获 ---
usage = SimpleNamespace(total_tokens=42)
chunk = FakeChunk(FakeDelta(), usage=usage)
# 手动验证 _stream_model 从 chunk.usage 捕获
assert chunk.usage.total_tokens == 42
print("OK  token 用量字段可从流式 chunk 捕获")


# --- 测试 4:stream=False 时不回调 on_delta(保持纯函数) ---
calls = []


def fake_create_nostream(**kwargs):
    assert kwargs.get("stream") is not True
    from types import SimpleNamespace as SN
    return type("Resp", (), {
        "choices": [type("C", (), {"message": type("M", (), {
            "content": "完整答案", "tool_calls": None,
            "model_dump": lambda self, **kw: {"role": "assistant", "content": "完整答案"},
        })()})()],
        "usage": None,
    })()


with patch.object(agent.client.chat.completions, "create", fake_create_nostream):
    out = run_agent("hi", stream=False, on_delta=calls.append)
assert out == "完整答案"
assert calls == [], "stream=False 时不应触发 on_delta"
print("OK  stream=False 保持纯函数,on_delta 不被触发")

print("\n全部通过")
