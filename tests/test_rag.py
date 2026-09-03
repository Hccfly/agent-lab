"""RAG 层冒烟测试:不调真实 API,monkeypatch 模拟 embedding 和模型。"""
import sys
from pathlib import Path

# 让脚本能从项目根目录导入模块(agent/rag 等)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import tempfile
from unittest.mock import patch

import agent
from agent import run_agent
from rag import VectorStore


# ---------- 测试 1:向量库检索(本地逻辑,不需 API) ----------
with tempfile.TemporaryDirectory() as td:
    vs = VectorStore(os.path.join(td, "db.json"))
    vs.add("关于Python的记忆", [1.0, 0.0, 0.0])
    vs.add("关于猫的记忆", [0.0, 1.0, 0.0])
    vs.save()

    vs2 = VectorStore(os.path.join(td, "db.json"))
    assert len(vs2) == 2, "持久化 round-trip 失败"
    hits = vs2._scores([1.0, 0.0, 0.0])
    assert hits[0][0] == 1.0 and hits[1][0] == 0.0, f"余弦相似度计算错误: {hits}"
    print("OK  向量库:增删/持久化/余弦相似度")


# ---------- 测试 2:用 stub store 验证 run_agent 的检索注入与入库 ----------
class StubStore:
    def __init__(self, path):
        self.added = []

    def search(self, query, top_k=3):
        return [{"text": "用户: 我的名字是Alice", "score": 1.0}]

    def add(self, text, vector):
        self.added.append(text)

    def save(self):
        pass


class FakeMessage:
    def __init__(self, content=None):
        self.content = content
        self.tool_calls = None

    def model_dump(self, mode="python", exclude_none=True):
        return {"role": "assistant", "content": self.content}


class FakeResp:
    def __init__(self, message):
        self.choices = [type("C", (), {"message": message})()]


def fake_create(**kwargs):
    system = kwargs["messages"][0]["content"]
    agent._last_system = system
    return FakeResp(FakeMessage(content="OK"))


sid = "rag-test-1"
store = StubStore("unused")
with patch("agent.VectorStore", lambda p: store), \
     patch.object(agent.client.chat.completions, "create", fake_create):
    out = run_agent("我叫什么名字?", session_id=sid)

assert out == "OK"
# 检索到的历史被注入 system
assert "我的名字是Alice" in agent._last_system, f"RAG 注入失败: {agent._last_system!r}"
print("OK  RAG 注入:system 包含检索到的历史")

# 本轮内容被入库
assert any("用户: 我叫什么名字?" in t for t in store.added), f"入库失败: {store.added}"
print(f"OK  入库: 向量库记录了本轮 {len(store.added)} 条")

print("\n全部通过")
