"""HTTP 服务层冒烟测试:不调真实 API,monkeypatch 掉 run_agent。
覆盖:GET /、POST /chat 一次性、POST /chat/stream 的 SSE 逐块推送 + session_id 事件。
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import server

client = TestClient(server.app)


# --- 测试 1:GET / 返回服务说明 JSON ---
r = client.get("/")
assert r.status_code == 200
assert "POST /chat/stream" in r.json()["endpoints"]
print("OK  GET / 返回服务说明")


# --- 测试 2:POST /chat 一次性返回答案与 session_id ---
def fake_run_sync(question, session_id=None, **kwargs):
    return f"回答[{session_id}]: 现在几点"


with patch("server.run_agent", side_effect=fake_run_sync):
    r = client.post("/chat", json={"question": "现在几点"})

assert r.status_code == 200
data = r.json()
assert data["session_id"], "服务端应生成 session_id"
assert "现在几点" in data["answer"]
print("OK  POST /chat:返回答案 + 自动生成 session_id")


# --- 测试 3:POST /chat 显式传 session_id -> 复用,不新生成 ---
with patch("server.run_agent", side_effect=fake_run_sync):
    r = client.post("/chat", json={"question": "hi", "session_id": "abc-1"})

data = r.json()
assert data["session_id"] == "abc-1", "显式传 session_id 应被原样使用"
assert "回答[abc-1]" in data["answer"]
print("OK  POST /chat:显式 session_id 被复用(续聊/多客户端隔离)")


# --- 测试 4:POST /chat/stream SSE:逐块推送 + [DONE] + session 事件 ---
def fake_run_stream(question, session_id=None, stream=True, on_delta=None, **kwargs):
    # 模拟模型逐字输出,on_delta 会被逐块调用
    for ch in ["你", "好", ",", "agent"]:
        on_delta(ch)
    return "你好,agent"


with patch("server.run_agent", side_effect=fake_run_stream):
    with client.stream("POST", "/chat/stream", json={"question": "打招呼"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        text = "".join(resp.iter_text())

assert "event: session" in text, text
assert "data: 你" in text and "data: 好" in text and "data: ," in text and "data: agent" in text, text
assert "data: [DONE]" in text, text
print("OK  POST /chat/stream:SSE 逐块推送 content,结尾 [DONE],开头 session 事件")


print("\n全部通过")
