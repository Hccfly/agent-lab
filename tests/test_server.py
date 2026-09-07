"""HTTP 服务层冒烟测试：结构化 SSE、异常收尾与 LangGraph 轨迹流。"""
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


# --- 测试 4:危险 session_id 在 HTTP 参数层直接返回 422 ---
with patch("server.run_agent", side_effect=fake_run_sync) as mocked:
    r = client.post("/chat", json={"question": "hi", "session_id": "../escape"})

assert r.status_code == 422
mocked.assert_not_called()
print("OK  POST /chat:路径穿越 session_id 在执行 Agent 前被拒绝")


# --- 测试 5:POST /chat/stream SSE:所有 payload 都是结构化 JSON ---
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
assert text.count("event: delta") == 4, text
assert 'data: {"text": "你"}' in text and 'data: {"text": "agent"}' in text, text
assert "event: done" in text and '"status": "completed"' in text, text
print("OK  POST /chat/stream:session/delta/done 均为结构化 JSON")


# --- 测试 6:后台异常不再被 finally 吞掉，而是 error + failed done ---
def fake_run_stream_error(*_args, **_kwargs):
    raise RuntimeError("上游模型断开")


with patch("server.run_agent", side_effect=fake_run_stream_error):
    with client.stream("POST", "/chat/stream", json={"question": "触发异常"}) as resp:
        error_text = "".join(resp.iter_text())

assert "event: error" in error_text, error_text
assert '"type": "RuntimeError"' in error_text, error_text
assert '"message": "上游模型断开"' in error_text, error_text
assert "event: done" in error_text and '"status": "failed"' in error_text, error_text
print("OK  POST /chat/stream:后台异常可解析且流正常结束")


# --- 测试 7:/graph/stream 复用观察器协议，输出 trace/result/done ---
def fake_observed(input_text, mode="single", observer=None, **_kwargs):
    observer.start(mode=mode, input_text=input_text)
    span_id, started, started_ms = observer.node_start("model", {"iteration": 0})
    observer.node_end(
        "model",
        span_id,
        started,
        started_ms,
        {"answer": "离线答案", "stop_reason": "completed"},
    )
    result = {"answer": "离线答案", "stop_reason": "completed"}
    observer.finish("completed", result=result)
    return result, observer


with patch("server.run_observed_graph", side_effect=fake_observed):
    with client.stream(
        "POST",
        "/graph/stream",
        json={"question": "展示轨迹", "mode": "single"},
    ) as resp:
        graph_text = "".join(resp.iter_text())

assert "event: run" in graph_text, graph_text
assert "event: trace" in graph_text and '"event": "node_end"' in graph_text, graph_text
assert '"duration_ms"' in graph_text and '"state_delta"' in graph_text, graph_text
assert "event: result" in graph_text and '"answer": "离线答案"' in graph_text, graph_text
assert "event: done" in graph_text and '"status": "completed"' in graph_text, graph_text
print("OK  POST /graph/stream:节点、状态、耗时、结果与结束事件完整")


# --- 测试 8:/graph/stream 异常同样输出结构化 error ---
def fake_observed_error(*_args, observer=None, **_kwargs):
    observer.emit(
        "node_error",
        node="planner",
        status="failed",
        error={"type": "ValueError", "message": "规划失败"},
    )
    raise ValueError("规划失败")


with patch("server.run_observed_graph", side_effect=fake_observed_error):
    with client.stream(
        "POST",
        "/graph/stream",
        json={"question": "坏规划", "mode": "multi"},
    ) as resp:
        graph_error_text = "".join(resp.iter_text())

assert "event: trace" in graph_error_text and '"node": "planner"' in graph_error_text
assert "event: error" in graph_error_text and '"message": "规划失败"' in graph_error_text
assert "event: done" in graph_error_text and '"status": "failed"' in graph_error_text
print("OK  POST /graph/stream:失败节点与传输错误均可解析")


print("\n全部通过")
