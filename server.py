"""HTTP 服务化：REST 一次性回答 + 结构化 SSE 文本流/图运行轨迹。

设计要点:
- run_agent 本质是"输入字符串 -> 输出字符串",会话靠 session_id 参数。
  服务化 = 加一层网络壳:核心循环 / 工具 / 记忆 / RAG 一行不改。
  同一个 run_agent 两种姿态:stream=False 一次性返回 JSON;
  stream=True + 换 on_delta -> SSE 逐块推送(关注点分离的红利)。
- SSE 为什么有必要:普通 HTTP 一次性返回整个 body,模型响应几十秒前端就
  白等几十秒。SSE(Server-Sent Events)让服务端把 on_delta 的每个块主动
  推给客户端,网页也能"打字机"。单向推送正好贴"模型逐字吐"的场景；
  POST 流可由浏览器 fetch + ReadableStream 消费。
- 流式难点:run_agent 是同步阻塞(内部 while 循环),HTTP 响应却要"边生成
  边吐"。做法:run_agent 放后台线程跑,on_delta 把块投进 asyncio.Queue,
  async 生成器从队列取出逐块 yield —— "同步核心 + 异步传输"的标准缝合。

接口:
    GET  /                       根路径:JSON 服务说明
    POST /chat                   一次性 JSON
    POST /chat/stream            session/delta/error/done 结构化 SSE
    POST /graph/stream           run/trace/result/error/done 结构化 SSE

请求体: {"question": "...", "session_id": "可选;不传则由服务端生成并返回}
会话:传同一个 session_id 就能跨请求续聊(memory 层本来就支持)。

启动:
    python -m uvicorn server:app --reload --port 8000
    curl -X POST localhost:8000/chat -H "content-type: application/json" -d '{"question":"现在几点"}'
    curl -N -X POST localhost:8000/graph/stream -H "content-type: application/json" -d '{"question":"计算 1+2","mode":"single"}'
"""
import asyncio
import datetime
import json
import threading
import uuid
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from agent import run_agent
from memory import validate_session_id
from observability import RunObserver, run_observed_graph, serialize_error

app = FastAPI(title="最小 Agent HTTP 服务", version="0.1.0")

# 宽松 CORS:允许任意源的跨源调用(未来若接独立前端/第三方客户端)。
# 上生产换成白名单 + 鉴权。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatBody(BaseModel):
    question: str
    session_id: str | None = None

    @field_validator("session_id")
    @classmethod
    def session_id_must_be_safe(cls, value: str | None) -> str | None:
        return validate_session_id(value) if value is not None else None


class GraphBody(BaseModel):
    question: str = Field(min_length=1)
    mode: Literal["single", "multi"] = "single"
    max_iterations: int = Field(default=10, ge=1)
    max_tasks: int = Field(default=5, ge=1)
    max_concurrency: int = Field(default=4, ge=1)


def _new_session_id() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


@app.get("/")
def index():
    return {
        "service": "agent-lab HTTP",
        "endpoints": {
            "POST /chat": "一次性 JSON:{question, session_id?} -> {session_id, answer}",
            "POST /chat/stream": "结构化 SSE:session/delta/error/done",
            "POST /graph/stream": "LangGraph 运行轨迹 SSE:节点/状态/耗时/失败位置",
        },
    }


@app.post("/chat")
def chat(body: ChatBody):
    """一次性:整个回答一个 JSON 返回。适合多 agent 汇总等不适合流式的结果。"""
    session_id = body.session_id or _new_session_id()
    answer = run_agent(body.question, session_id=session_id, stream=False)
    return {"session_id": session_id, "answer": answer}


def _sse(event: str, data: dict[str, Any]) -> str:
    """每条 SSE 都使用事件名 + 单行 JSON，文本中的换行不会破坏帧边界。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _event_stream(body: ChatBody):
    """SSE 生成器。

    后台线程跑 run_agent(stream=True),on_delta 把每个增量块投进
    asyncio.Queue;这里(事件循环内)从队列取出逐块 yield。线程把响应
    推送跨线程调度到本请求的 loop,避免阻塞事件循环。
    """
    session_id = body.session_id or _new_session_id()
    loop = asyncio.get_running_loop()
    q: "asyncio.Queue[tuple[str, dict[str, Any]] | object]" = asyncio.Queue()
    done_sentinel = object()

    def enqueue(event: str, data: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(q.put_nowait, (event, data))

    def on_delta(text: str) -> None:
        enqueue("delta", {"text": text})

    def worker() -> None:
        status = "completed"
        try:
            run_agent(body.question, session_id=session_id, stream=True, on_delta=on_delta)
        except Exception as exc:
            status = "failed"
            enqueue("error", serialize_error(exc))
        finally:
            enqueue("done", {"status": status, "session_id": session_id})
            loop.call_soon_threadsafe(q.put_nowait, done_sentinel)

    # 先把生效的 session_id 作为独立事件推给前端(便于续聊/关联日志)
    yield _sse("session", {"session_id": session_id})
    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = await q.get()
        if item is done_sentinel:
            break
        event, data = item
        yield _sse(event, data)


@app.post("/chat/stream")
async def chat_stream(body: ChatBody):
    """SSE 流式:逐字把 run_agent 的 on_delta 增量推给客户端(网页打字机)。"""
    return StreamingResponse(
        _event_stream(body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _graph_event_stream(body: GraphBody):
    """把统一 RunObserver 事件逐条转成 SSE，失败也以 JSON 事件正常收尾。"""
    loop = asyncio.get_running_loop()
    q: "asyncio.Queue[tuple[str, dict[str, Any]] | object]" = asyncio.Queue()
    done_sentinel = object()
    run_id = uuid.uuid4().hex

    def enqueue(event: str, data: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(q.put_nowait, (event, data))

    observer = RunObserver(
        run_id=run_id,
        on_event=lambda event: enqueue("trace", event),
    )

    def worker() -> None:
        status = "completed"
        try:
            result, _ = run_observed_graph(
                body.question,
                mode=body.mode,
                observer=observer,
                max_iterations=body.max_iterations,
                max_tasks=body.max_tasks,
                max_concurrency=body.max_concurrency,
            )
            enqueue(
                "result",
                {
                    "answer": result.get("answer", ""),
                    "stop_reason": result.get("stop_reason", ""),
                },
            )
        except Exception as exc:
            status = "failed"
            enqueue("error", serialize_error(exc))
        finally:
            enqueue("done", {"run_id": run_id, "status": status})
            loop.call_soon_threadsafe(q.put_nowait, done_sentinel)

    yield _sse("run", {"run_id": run_id, "mode": body.mode})
    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = await q.get()
        if item is done_sentinel:
            break
        event, data = item
        yield _sse(event, data)


@app.post("/graph/stream")
async def graph_stream(body: GraphBody):
    """实时返回 LangGraph 节点、状态增量、耗时、结果与失败位置。"""
    return StreamingResponse(
        _graph_event_stream(body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
