"""核心:最小可用的 agent 循环 + 会话状态/记忆 + 流式输出。

整个 agent 的本质就是下面这个 while 循环:
    调模型 -> 模型说"我要调工具" -> 执行工具 -> 把结果回填给模型
             -> 再调模型 -> ... -> 模型说"我有最终答案了" -> 结束

状态与记忆:
- 每次调用传入 session_id,对话历史持久化到 sessions/<id>.json;
  下次用同一 id 调用,能接着上次继续聊(短期记忆)。
- 历史过长时,早期消息被压缩成摘要放进 system prompt(长期记忆),
  从而控制 token 成本,同时保留关键事实。

流式输出:
- run_agent(stream=True, on_delta=...):模型输出逐字回调 on_delta;
  stream=False(默认)保持纯函数——返回完整答案,便于测试。
"""
import json
import logging
import os
import time
from types import SimpleNamespace

from dotenv import load_dotenv
from openai import OpenAI

# 必须在读取超时/并发配置的本地模块导入前加载 .env，避免配置依赖导入顺序。
load_dotenv(override=True)

import memory
from log import log_event
from rag import VectorStore, embed_texts
from tools import IDEMPOTENT_TOOLS, TOOLS, run_tool, tool_key
from governance import env_positive_float

client = OpenAI()  # 自动读取环境变量 OPENAI_API_KEY / OPENAI_BASE_URL
MODEL = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")
MODEL_TIMEOUT_SECONDS = env_positive_float("AGENT_MODEL_TIMEOUT_SECONDS", 30.0)

SYSTEM_PROMPT = """你是一个能调用工具的小助手。
判断用户的问题是否需要工具:
- 需要实时信息或精确计算时,调用对应工具;
- 不要编造工具的返回结果,只基于工具实际返回的内容组织回答。
回答使用简体中文,简洁清晰。"""

# 合流模式提示:让模型在调用工具前先显式输出思考(Thought)。
# 设计要点:Reasoning(Thought)与 Action(tool_calls)是两条独立通道——
# 能输出 reasoning_content 的模型(如 DeepSeek-reasoner、Qwen thinking)
# 把 Thought 放进独立字段,Action 走结构化 tool_calls,两者并存互不干扰。
REASONING_HINT = (
    "\n\n在调用任何工具之前,先简短说明你的推理(Thought),"
    "例如:需要实时时间,因此调用 get_current_time。"
    "注意:推理是给你的思考过程,不要把它混入最终回答。"
)

# 上下文过长阈值:消息超过这么多条就触发压缩(记忆层)。
SUMMARY_THRESHOLD = 24
# 压缩时保留最近 KEEP_RECENT 条原始消息,其余转成摘要。
KEEP_RECENT = 8


def _build_messages(
    summary: str,
    convo: list[dict],
    related: list[str] | None = None,
    show_reasoning: bool = False,
    system_prompt: str | None = None,
) -> list[dict]:
    """把 [摘要 + 检索到的历史 + 对话历史] 拼成发给模型的完整消息列表。

    related: RAG 从向量库检索到的相关历史,供模型参考精确细节。
    show_reasoning: 合流模式,提示模型在工具调用前先输出 Thought。
    system_prompt: 覆盖默认角色(多专用 agent 时每个角色有自己的提示)。
    """
    system = system_prompt or SYSTEM_PROMPT
    if show_reasoning:
        system += REASONING_HINT
    if summary:
        system += f"\n\n# 之前的对话摘要\n{summary}"
    if related:
        system += "\n\n# 检索到的相关历史对话(供参考)\n" + "\n".join(related)
    return [{"role": "system", "content": system}] + convo


class _StreamedToolCall:
    """流式重建的"工具调用",提供与 SDK tool_call 一致的最小接口。"""

    __slots__ = ("id", "function")

    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)


class _StreamedMessage:
    """流式重建的助手消息,提供与 SDK message 一致的最小接口。"""

    def __init__(self, content: str, tool_calls: list, reasoning: str = ""):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning = reasoning

    def model_dump(self, mode="json", exclude_none=True) -> dict:
        d = {"role": "assistant"}
        if self.content:
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
        # 注意:reasoning 刻意不放进 model_dump——Thought 只用于展示和日志,
        # 不混入会话历史回填给模型(模型自己会再次生成 content 与 tool_calls)。
        return d


def _stream_model(messages: list[dict], on_delta=None, on_reasoning=None, tools: list | None = None):
    """流式调用模型,逐字回调 on_delta,重建消息对象并捕获 token 用量。

    设计要点:流式返回的每个 chunk 只带增量(delta)。content 直接拼接;
    tool_calls 按 index 分组累积——id / name / arguments 会分散在多个
    chunk 里,必须按 index 合并,否则工具调用信息不完整。

    on_reasoning: 合流模式。thinking 模型的推理(chunk.delta.reasoning_content)
    也是一路独立增量,用 getattr 防御性读取(非 thinking 模型没有该字段)。
    tools: 默认全量 TOOLS;多专用 agent 时传该角色的工具子集。
    """
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS if tools is None else tools,
        stream=True,
        stream_options={"include_usage": True},
        timeout=MODEL_TIMEOUT_SECONDS,
    )
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_map: dict[int, dict] = {}
    usage = None
    for chunk in resp:
        if chunk.usage is not None:
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            reasoning_parts.append(reasoning)
            if on_reasoning:
                on_reasoning(reasoning)
        if delta.content:
            content_parts.append(delta.content)
            if on_delta:
                on_delta(delta.content)
        for tc in (delta.tool_calls or []):
            entry = tool_map.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
            if tc.id:
                entry["id"] = tc.id
            if tc.function:
                if tc.function.name:
                    entry["name"] = tc.function.name
                if tc.function.arguments:
                    entry["arguments"] += tc.function.arguments
    tool_calls = [
        _StreamedToolCall(entry["id"], entry["name"], entry["arguments"])
        for _, entry in sorted(tool_map.items())
    ]
    return _StreamedMessage("".join(content_parts), tool_calls, "".join(reasoning_parts)), usage


def _fmt(m: dict) -> str:
    """把一条历史消息变成可读文本,供摘要模型阅读。"""
    role = m.get("role")
    if role == "tool":
        return f"[工具结果] {m.get('content', '')}"
    if m.get("tool_calls"):
        calls = "; ".join(tc["function"]["name"] for tc in m["tool_calls"])
        return f"[助手调用工具] {calls}"
    return f"{role}: {m.get('content', '')}"


def _compress(part: list[dict], prev_summary: str) -> str:
    """让模型把一段旧对话压成摘要;可基于已有摘要增量合并。

    注意:这个调用不传 tools,只让模型做纯文本总结。
    """
    prompt = (
        "下面是之前的一段 agent 对话记录。请用简体中文写一段不超过 200 字的摘要,"
        "保留:用户的关键诉求、已完成/待办的事项、工具调用得到的重要事实。"
        "如果没有任何实质内容,只回复\"无\"。\n"
    )
    if prev_summary:
        prompt += f"\n已有的历史摘要:\n{prev_summary}\n\n请在此基础上合并更新,不要遗漏旧事实。\n"
    prompt += "\n对话记录:\n" + "\n".join(_fmt(m) for m in part)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        timeout=MODEL_TIMEOUT_SECONDS,
    )
    return (resp.choices[0].message.content or "").strip()


def _store_round(store: VectorStore | None, user_input: str, answer: str, session_id: str | None = None) -> None:
    """把本轮对话切块后写入向量库,作为长期记忆供后续检索。

    失败(如 embedding 服务不可用)时静默跳过——记忆失败不应中断对话。
    """
    if store is None:
        return
    chunks = []
    if user_input.strip():
        chunks.append(f"用户: {user_input.strip()}")
    if answer.strip():
        chunks.append(f"助手: {answer.strip()}")
    if not chunks:
        return
    try:
        vectors = embed_texts(chunks)
        for text, vec in zip(chunks, vectors):
            store.add(text, vec)
        store.save()
        log_event(
            "rag_store",
            f"[rag] 已入库 {len(chunks)} 条(共 {len(store)} 条记忆)",
            added=len(chunks), total=len(store), session_id=session_id,
        )
    except Exception as exc:
        log_event("rag_error", f"[rag] 入库失败,跳过: {exc}", error=str(exc), session_id=session_id)


def run_agent(
    user_input: str,
    session_id: str | None = None,
    max_iterations: int = 10,
    stream: bool = False,
    on_delta=None,
    show_reasoning: bool = False,
    on_reasoning=None,
    tools: list | None = None,
    system_prompt: str | None = None,
) -> str:
    """校验会话标识，并将同一 session 的完整读改写事务串行化。"""
    if session_id is None:
        return _run_agent_unlocked(
            user_input, None, max_iterations, stream, on_delta,
            show_reasoning, on_reasoning, tools, system_prompt,
        )
    normalized = memory.validate_session_id(session_id)
    with memory.session_lock(normalized):
        return _run_agent_unlocked(
            user_input, normalized, max_iterations, stream, on_delta,
            show_reasoning, on_reasoning, tools, system_prompt,
        )


def _run_agent_unlocked(
    user_input: str,
    session_id: str | None = None,
    max_iterations: int = 10,
    stream: bool = False,
    on_delta=None,
    show_reasoning: bool = False,
    on_reasoning=None,
    tools: list | None = None,
    system_prompt: str | None = None,
) -> str:
    """核心循环。

    stream=True 时,模型输出逐字回调 on_delta(关注点分离:流式是
    展示层的事,默认 False 保持纯函数,便于测试和复用)。

    show_reasoning/on_reasoning: 合流模式。捕获模型的推理(Thought)——
    Reasoning 与 Action(tool_calls)是两条独立通道,Thought 只用于展示和
    日志,不混入会话历史回填给模型。

    tools/system_prompt: 多专用 agent 支持。默认全量工具 + 默认角色;
    专用 executor 传入自己的工具子集和角色提示(能力边界)。
    """
    effective_tools = TOOLS if tools is None else tools
    state = memory.load_session(session_id) if session_id else {"summary": "", "messages": []}
    # convo 是本轮的工作历史(不含 system):历史 + 本次提问
    convo = list(state["messages"]) + [{"role": "user", "content": user_input}]

    # RAG:用本次提问检索相关历史。向量库按会话独立持久化。
    store = None
    related = []
    if session_id:
        store = VectorStore(str(memory.SESSION_DIR / f"{session_id}_vec.json"))
        try:
            hits = store.search(user_input, top_k=3)
            related = [h["text"] for h in hits]
            if hits:
                log_event(
                    "rag_retrieve",
                    "[rag] 检索到 %d 条历史" % len(hits),
                    hits=len(hits),
                    top_scores=[h["score"] for h in hits],
                    session_id=session_id,
                )
            else:
                log_event("rag_retrieve", "[rag] 向量库为空,无检索结果", hits=0, session_id=session_id)
        except Exception as exc:
            # 检索失败(如 embedding 服务不可用)不应中断对话,降级为不注入。
            log_event("rag_error", f"[rag] 检索失败,降级: {exc}", error=str(exc), session_id=session_id)
            related = []
    else:
        related = []

    # 本轮工具结果缓存:幂等工具的相同调用只真执行一次,后续直接复用。
    # 生命周期限定在单次 run_agent 内(跨轮复用是"缓存"而非"幂等去重",
    # 且对非幂等工具如时间有危险)。仅成功结果入缓存,报错不缓存以鼓励重试。
    dedup_cache: dict[str, str] = {}

    for _ in range(max_iterations):
        # 记忆层:上下文过长时,把早期消息压缩成摘要,只留最近的原文。
        if len(convo) > SUMMARY_THRESHOLD:
            state["summary"] = _compress(convo[:-KEEP_RECENT], state["summary"])
            convo = convo[-KEEP_RECENT:]

        messages = _build_messages(
            state["summary"], convo, related, show_reasoning, system_prompt=system_prompt
        )
        if stream:
            message, usage = _stream_model(
                messages, on_delta=on_delta, on_reasoning=on_reasoning, tools=effective_tools
            )
        else:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=effective_tools,
                timeout=MODEL_TIMEOUT_SECONDS,
            )
            message = response.choices[0].message
            usage = getattr(response, "usage", None)
        if usage is not None:
            log_event(
                "llm_call", f"[llm] 本轮 tokens={usage.total_tokens}",
                tokens=usage.total_tokens, session_id=session_id,
            )

        # 合流:捕获模型推理(Thought)。流式重建的 message 存 reasoning 属性,
        # 非流式 SDK message 是 reasoning_content 字段(getattr 防御)。
        reasoning = getattr(message, "reasoning", None)
        if reasoning is None:
            reasoning = getattr(message, "reasoning_content", None) or ""
        if show_reasoning and reasoning:
            log_event(
                "agent_reasoning", "[thinking] " + reasoning[:80],
                reasoning=reasoning[:500], session_id=session_id,
            )

        # 终止条件:模型本轮没有发起任何工具调用,说明要输出最终回答了。
        if not message.tool_calls:
            answer = message.content or ""
            convo.append({"role": "assistant", "content": answer})
            if session_id:
                memory.save_session(session_id, state["summary"], convo)
                _store_round(store, user_input, answer, session_id)
            log_event("agent_answer", "", length=len(answer), session_id=session_id)
            return answer

        # 把模型的对象转成纯 dict,既方便持久化,也能原样回传给模型。
        # pop reasoning_content:合流模式下非流式 SDK message 可能带该字段,
        # 但 Thought 不该混入会话历史回填(模型会重新生成 content/tool_calls)。
        assistant_msg = message.model_dump(mode="json", exclude_none=True)
        assistant_msg.pop("reasoning_content", None)
        convo.append(assistant_msg)

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args_raw = tool_call.function.arguments
            log_event("tool_call", f"[tool] {name}({args_raw})", tool=name, arguments=args_raw, session_id=session_id)
            args = json.loads(args_raw or "{}")
            # 幂等工具命中缓存 -> 不重新执行,直接复用上次结果回填给模型。
            key = tool_key(name, args) if name in IDEMPOTENT_TOOLS else None
            ok = False
            if key is not None and key in dedup_cache:
                result = dedup_cache[key]
                ok = True
                log_event(
                    "tool_result",
                    f"[tool] {name} -> (幂等去重:复用上次结果) {result[:60]}",
                    tool=name, ok=True, dedup=True,
                    latency_ms=0.0, session_id=session_id,
                )
            else:
                t0 = time.monotonic()
                try:
                    result = run_tool(name, args)
                    ok = True
                    log_event(
                        "tool_result", f"[tool] {name} -> {result[:60]}",
                        tool=name, ok=True,
                        latency_ms=round((time.monotonic() - t0) * 1000, 1),
                        session_id=session_id,
                    )
                except Exception as exc:
                    # 工具报错时,把错误信息回填给模型,让它自己决定下一步。
                    # 设计要点:这比代码里硬编码重试更符合 agent 的行为逻辑。
                    # 报错不写入缓存:同参重试仍可能成功,且鼓励模型换参数。
                    result = f"工具执行出错: {exc}"
                    log_event(
                        "tool_result", f"[tool] {name} 出错: {exc}",
                        tool=name, ok=False, error=str(exc),
                        latency_ms=round((time.monotonic() - t0) * 1000, 1),
                        session_id=session_id,
                    )
            if ok and key is not None:
                dedup_cache[key] = result
            convo.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    # 兜底:防止极端情况下模型陷入工具调用死循环。
    fallback = f"达到最大迭代次数({max_iterations})仍未结束,已强制停止。"
    if on_delta:
        on_delta(fallback)
    return fallback
