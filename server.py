"""HTTP 服务化:用 FastAPI 包住 CLI 入口,提供 REST 一次性 + SSE 流式两种接口。

设计要点:
- run_agent 本质是"输入字符串 -> 输出字符串",会话靠 session_id 参数。
  服务化 = 加一层网络壳:核心循环 / 工具 / 记忆 / RAG 一行不改。
  同一个 run_agent 两种姿态:stream=False 一次性返回 JSON;
  stream=True + 换 on_delta -> SSE 逐块推送(关注点分离的红利)。
- SSE 为什么有必要:普通 HTTP 一次性返回整个 body,模型想几十秒前端就
  白等几十秒。SSE(Server-Sent Events)让服务端把 on_delta 的每个块主动
  推给客户端,网页也能"打字机"。单向推送正好贴"模型逐字吐"的场景,
  浏览器原生支持 EventSource,前端几行代码就收到。
- 流式难点:run_agent 是同步阻塞(内部 while 循环),HTTP 响应却要"边生成
  边吐"。做法:run_agent 放后台线程跑,on_delta 把块投进 asyncio.Queue,
  async 生成器从队列取出逐块 yield —— "同步核心 + 异步传输"的标准缝合。

接口:
    GET  /                       根路径:JSON 服务说明
    POST /chat                   一次性 JSON
    POST /chat/stream            SSE 流式响应(text/event-stream)

请求体: {"question": "...", "session_id": "可选;不传则由服务端生成并返回}
会话:传同一个 session_id 就能跨请求续聊(memory 层本来就支持)。

启动:
    python -m uvicorn server:app --reload --port 8000
    curl -X POST localhost:8000/chat -H "content-type: application/json" -d '{"question":"现在几点"}'
    curl -N -X POST localhost:8000/chat/stream -H "content-type: application/json" -d '{"question":"现在几点"}'
"""
import asyncio
import datetime
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent import run_agent

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


def _new_session_id() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


@app.get("/")
def index():
    return {
        "service": "agent-lab HTTP",
        "endpoints": {
            "POST /chat": "一次性 JSON:{question, session_id?} -> {session_id, answer}",
            "POST /chat/stream": "SSE 流式(text/event-stream),curl -N 消费",
        },
    }


@app.post("/chat")
def chat(body: ChatBody):
    """一次性:整个回答一个 JSON 返回。适合多 agent 汇总等不适合流式的结果。"""
    session_id = body.session_id or _new_session_id()
    answer = run_agent(body.question, session_id=session_id, stream=False)
    return {"session_id": session_id, "answer": answer}


async def _event_stream(body: ChatBody):
    """SSE 生成器。

    后台线程跑 run_agent(stream=True),on_delta 把每个增量块投进
    asyncio.Queue;这里(事件循环内)从队列取出逐块 yield。线程把响应
    推送跨线程调度到本请求的 loop,避免阻塞事件循环。
    """
    session_id = body.session_id or _new_session_id()
    loop = asyncio.get_running_loop()
    q: "asyncio.Queue[str | None]" = asyncio.Queue()
    _DONE = None  # 哨兵

    def on_delta(text: str) -> None:
        asyncio.run_coroutine_threadsafe(q.put(text), loop)

    def worker() -> None:
        try:
            run_agent(body.question, session_id=session_id, stream=True, on_delta=on_delta)
        finally:
            # 结束信号:无论正常返回还是异常,都要让生成器收尾
            asyncio.run_coroutine_threadsafe(q.put(_DONE), loop)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # 先把生效的 session_id 作为独立事件推给前端(便于续聊/关联日志)
    yield f"event: session\ndata: {session_id}\n\n"
    while True:
        item = await q.get()
        if item is _DONE:
            break
        yield f"data: {item}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/chat/stream")
async def chat_stream(body: ChatBody):
    """SSE 流式:逐字把 run_agent 的 on_delta 增量推给客户端(网页打字机)。"""
    return StreamingResponse(
        _event_stream(body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
