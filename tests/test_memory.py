"""记忆层的冒烟测试:不调真实 API,用 monkeypatch 模拟模型。"""
import sys
from pathlib import Path

# 让脚本能从项目根目录导入模块(agent/memory 等)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import patch

import agent
import memory
from agent import run_agent


class StubStore:
    def __init__(self, _path):
        self.items = []

    def search(self, _query, top_k=3):
        return []

    def add(self, text, vector):
        self.items.append((text, vector))

    def save(self):
        pass

    def __len__(self):
        return len(self.items)


class FakeToolCall:
    def __init__(self, name, arguments, call_id="call_1"):
        self.id = call_id
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, mode="python", exclude_none=True):
        d = {"role": "assistant"}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in self.tool_calls
            ]
        return d


class FakeResp:
    def __init__(self, message):
        self.choices = [type("C", (), {"message": message})()]


def fake_create(**kwargs):
    msgs = kwargs.get("messages", [])
    # 特征:压缩请求 = 单条 user 消息、不带 tools
    if not kwargs.get("tools") and len(msgs) == 1 and "对话记录" in msgs[0]["content"]:
        return FakeResp(FakeMessage(content="压缩得到的摘要"))
    # 工具循环:只要历史里还没有 tool_calls,就先让它调一次 get_current_time
    if not any(m.get("tool_calls") for m in msgs):
        return FakeResp(FakeMessage(tool_calls=[FakeToolCall("get_current_time", "{}")]))
    return FakeResp(FakeMessage(content="当前时间测试值"))


sid = "test-001"

# --- 测试 1:会话存取 round-trip ---
memory.save_session(sid, "摘要A", [{"role": "user", "content": "你好"}])
st = memory.load_session(sid)
assert st["summary"] == "摘要A" and len(st["messages"]) == 1
print("OK  会话存取 round-trip")

# --- 测试 2:循环 + 持久化 ---
with patch.object(agent.client.chat.completions, "create", fake_create), \
     patch.object(agent, "VectorStore", StubStore), \
     patch.object(agent, "embed_texts", side_effect=lambda texts: [[1.0] for _ in texts]):
    out = run_agent("现在几点", session_id=sid)
assert out == "当前时间测试值"
st = memory.load_session(sid)
roles = [m["role"] for m in st["messages"]]
# 第一条 user 来自测试 1 预置的会话,证明跨次调用记忆生效
assert roles == ["user", "user", "assistant", "tool", "assistant"], roles
print("OK  循环与持久化,历史 roles:", roles)

# --- 测试 3:压缩 ---
agent.SUMMARY_THRESHOLD = 4
agent.KEEP_RECENT = 2
with patch.object(agent.client.chat.completions, "create", fake_create), \
     patch.object(agent, "VectorStore", StubStore), \
     patch.object(agent, "embed_texts", side_effect=lambda texts: [[1.0] for _ in texts]):
    out = run_agent("再问一次", session_id=sid)
st = memory.load_session(sid)
assert st["summary"], "应当生成摘要"
# 压缩把历史从 6 条裁到 2 条,新一轮工具调用又产生 3 条 -> 5 条;
# 不压缩的话本轮会涨到 9 条。断言 <=5 即证明压缩生效、历史没有无限膨胀。
assert len(st["messages"]) <= 5, f"压缩后历史应被裁剪,当前 {len(st['messages'])}"
print(f"OK  压缩后: 摘要='{st['summary']}' 历史条数={len(st['messages'])}")

print("\n全部通过")
