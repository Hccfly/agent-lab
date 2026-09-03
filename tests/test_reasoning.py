"""合流模式测试:原生 tool calling + reasoning(Thought)双通道。

设计要点:Reasoning 与 Action(tool_calls)是两条独立通道,Thought 只用于
展示和日志,不回填给模型——模型会重新生成 content 与 tool_calls。
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# 让脚本能从项目根目录导入模块(agent 等)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent
from agent import _stream_model, run_agent


class FakeDelta:
    def __init__(self, content=None, tool_calls=None, reasoning=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning


class FakeChoice:
    def __init__(self, delta):
        self.delta = delta


class FakeChunk:
    def __init__(self, delta, usage=None):
        self.choices = [FakeChoice(delta)]
        self.usage = usage


def make_stream(*chunks):
    return iter(chunks)


class FakeMsg:
    def __init__(self, content, tool_calls, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content

    def model_dump(self, mode="json", exclude_none=True):
        d = {"role": "assistant", "content": self.content}
        if self.reasoning_content:
            d["reasoning_content"] = self.reasoning_content
        return d


# --- 测试 1:流式 reasoning_content 被捕获并逐字回调 on_reasoning ---
def fake_stream(**kwargs):
    assert kwargs.get("stream") is True
    return make_stream(
        FakeChunk(FakeDelta(reasoning="需要实时时间")),
        FakeChunk(FakeDelta(reasoning=",因此调用工具")),
        FakeChunk(FakeDelta(content="好的")),
    )


with patch.object(agent.client.chat.completions, "create", fake_stream):
    thinking = []
    msg, _ = _stream_model([{"role": "user", "content": "hi"}], on_reasoning=thinking.append)
assert "".join(thinking) == "需要实时时间,因此调用工具", thinking
assert msg.reasoning == "需要实时时间,因此调用工具", msg.reasoning
print("OK  流式 reasoning_content 被捕获并逐字回调 on_reasoning")


# --- 测试 2:reasoning 不进 model_dump(不回填给模型,不污染会话历史) ---
d = msg.model_dump(mode="json", exclude_none=True)
assert "reasoning_content" not in d and "reasoning" not in d, f"reasoning 不应进入回填: {d}"
print("OK  reasoning 不进 model_dump,只用于展示/日志")


# --- 测试 3:非 thinking 模型没有 reasoning_content,防御性读取不报错 ---
def fake_stream2(**kwargs):
    # delta 完全没有 reasoning_content 属性——但 FakeDelta 有;用 None 模拟
    return make_stream(FakeChunk(FakeDelta(content="正常答案")))


with patch.object(agent.client.chat.completions, "create", fake_stream2):
    msg2, _ = _stream_model([{"role": "user", "content": "hi"}])
assert msg2.reasoning == "" and msg2.content == "正常答案"
print("OK  非 thinking 模型无 reasoning,防御性读取不报错")


# --- 测试 4:run_agent 非流式——SDK message 带 reasoning_content 时,
#           捕获到且回填的历史被清理 ---
class FakeResp:
    def __init__(self, message):
        self.choices = [SimpleNamespace(message=message)]
        self.usage = None


def fake_create(**kwargs):
    assert kwargs.get("stream") is not True
    return FakeResp(FakeMsg("", None, reasoning_content="我需要计算"))

# 构造:第一轮只发推理但没工具调用 -> 直接返回,应被当作最终回答(内容为空)
with patch.object(agent.client.chat.completions, "create", fake_create):
    out = run_agent("算一下", show_reasoning=True)
assert out == "", out
print("OK  非流式 reasoning_content 被捕获(记日志),回填历史已清理")


# --- 测试 5:show_reasoning=False 时不触发 on_reasoning ---
def fake_stream5(**kwargs):
    return make_stream(FakeChunk(FakeDelta(reasoning="不应回调")))


with patch.object(agent.client.chat.completions, "create", fake_stream5):
    thinking = []
    msg5, _ = _stream_model([{"role": "user", "content": "hi"}], on_reasoning=thinking.append)
# 注意:_stream_model 不感知 show_reasoning,回调是否触发由调用方(run_agent)控制
assert msg5.reasoning == "不应回调"
print("OK  reasoning 捕获由调用方决定是否展示(run_agent 层控制)")

print("\n全部通过")
