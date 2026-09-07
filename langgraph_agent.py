"""LangGraph 版单 Agent：把手写 while 循环改造成显式状态图。

这不是对 ``agent.py`` 的替代，而是同一套模型与工具层的第二种编排实现。
保留两版可以直接对比手写控制流与框架状态图。

图结构：
    START -> model --无工具调用--> END
                    --有工具调用--> tools -> model
                                      └─达到上限--> force_stop -> END
"""
from __future__ import annotations

import json
import operator
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent import MODEL, MODEL_TIMEOUT_SECONDS, SYSTEM_PROMPT, client
from observability import RunObserver, observe_node
from tools import IDEMPOTENT_TOOLS, TOOLS, run_tool, tool_key


class PendingToolCall(TypedDict):
    """脱离 SDK 对象后的可序列化工具调用。"""

    id: str
    name: str
    arguments: str


class TraceEvent(TypedDict, total=False):
    """用于调试、测试和后续评测的最小轨迹事件。"""

    node: str
    iteration: int
    tool_calls: list[str]
    tool: str
    ok: bool
    dedup: bool
    approval: str
    arguments: dict[str, Any]
    detail: str


class AgentState(TypedDict):
    """单 Agent 的完整运行时状态。

    ``messages`` 和 ``trace`` 使用 reducer：节点只返回本次新增内容，
    LangGraph 负责追加。其他字段使用默认覆盖语义。
    """

    messages: Annotated[list[dict[str, Any]], operator.add]
    pending_tool_calls: list[PendingToolCall]
    tool_cache: dict[str, str]
    iteration: int
    max_iterations: int
    answer: str
    stop_reason: str
    trace: Annotated[list[TraceEvent], operator.add]


@dataclass(frozen=True)
class AgentDependencies:
    """图节点依赖，显式注入后可完全离线测试。"""

    model_client: Any
    model: str
    tool_schemas: list[dict]
    tool_runner: Callable[[str, dict], str]
    system_prompt: str
    idempotent_tools: frozenset[str]
    sensitive_tools: frozenset[str] = frozenset()
    request_timeout_seconds: float = 30.0


def default_dependencies() -> AgentDependencies:
    return AgentDependencies(
        model_client=client,
        model=MODEL,
        tool_schemas=TOOLS,
        tool_runner=run_tool,
        system_prompt=SYSTEM_PROMPT,
        idempotent_tools=IDEMPOTENT_TOOLS,
        sensitive_tools=frozenset(),
        request_timeout_seconds=MODEL_TIMEOUT_SECONDS,
    )


def _serialize_assistant_message(message: Any) -> tuple[dict[str, Any], list[PendingToolCall]]:
    """把 OpenAI SDK 消息转为 JSON 可序列化 dict，并抽取待执行调用。"""
    pending: list[PendingToolCall] = []
    serialized_calls = []
    for call in message.tool_calls or []:
        item: PendingToolCall = {
            "id": call.id,
            "name": call.function.name,
            "arguments": call.function.arguments or "{}",
        }
        pending.append(item)
        serialized_calls.append(
            {
                "id": item["id"],
                "type": "function",
                "function": {
                    "name": item["name"],
                    "arguments": item["arguments"],
                },
            }
        )

    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": message.content or "",
    }
    if serialized_calls:
        assistant_message["tool_calls"] = serialized_calls
    return assistant_message, pending


def _normalize_approval_decisions(
    response: Any,
    sensitive_calls: list[PendingToolCall],
) -> dict[str, dict[str, Any]]:
    """严格校验一次 interrupt 的整批人工决策。"""
    if not isinstance(response, dict) or not isinstance(response.get("decisions"), list):
        raise ValueError("审批响应必须是包含 decisions 数组的 JSON 对象")

    expected = {call["id"] for call in sensitive_calls}
    normalized: dict[str, dict[str, Any]] = {}
    for item in response["decisions"]:
        if not isinstance(item, dict):
            raise ValueError("每项审批决策必须是 JSON 对象")
        call_id = item.get("call_id")
        action = item.get("action")
        if call_id not in expected:
            raise ValueError(f"未知的待审批 call_id: {call_id!r}")
        if call_id in normalized:
            raise ValueError(f"call_id {call_id!r} 出现重复决策")
        if action not in {"approve", "reject", "edit"}:
            raise ValueError(f"不支持的审批动作: {action!r}")
        decision: dict[str, Any] = {"action": action}
        if action == "edit":
            arguments = item.get("arguments")
            if not isinstance(arguments, dict):
                raise ValueError("edit 决策必须提供 JSON 对象 arguments")
            decision["arguments"] = arguments
        if action == "reject":
            reason = item.get("reason", "用户未批准该操作")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("reject 决策的 reason 必须是非空字符串")
            decision["reason"] = reason.strip()
        normalized[call_id] = decision

    missing = expected - normalized.keys()
    if missing:
        raise ValueError(f"以下敏感调用缺少审批决策: {sorted(missing)}")
    return normalized


def build_agent_graph(
    dependencies: AgentDependencies | None = None,
    checkpointer=None,
    observer: RunObserver | None = None,
    node_prefix: str = "",
):
    """构建并编译单 Agent 状态图。

    接受依赖注入而不是在测试里连接真实模型。后续增加 checkpoint 时，
    只需给 ``builder.compile(checkpointer=...)`` 传入持久化器。
    """
    deps = dependencies or default_dependencies()

    def model_node(state: AgentState) -> dict:
        response = deps.model_client.chat.completions.create(
            model=deps.model,
            messages=[{"role": "system", "content": deps.system_prompt}] + state["messages"],
            tools=deps.tool_schemas,
            timeout=deps.request_timeout_seconds,
        )
        message = response.choices[0].message
        assistant_message, pending = _serialize_assistant_message(message)
        iteration = state["iteration"] + 1
        answer = assistant_message["content"] if not pending else ""
        stop_reason = "completed" if not pending else ""
        return {
            "messages": [assistant_message],
            "pending_tool_calls": pending,
            "iteration": iteration,
            "answer": answer,
            "stop_reason": stop_reason,
            "trace": [
                {
                    "node": "model",
                    "iteration": iteration,
                    "tool_calls": [call["name"] for call in pending],
                }
            ],
        }

    def tool_node(state: AgentState) -> dict:
        cache = dict(state["tool_cache"])
        tool_messages: list[dict[str, Any]] = []
        events: list[TraceEvent] = []

        sensitive_calls = [
            call for call in state["pending_tool_calls"]
            if call["name"] in deps.sensitive_tools
        ]
        decisions: dict[str, dict[str, Any]] = {}
        if sensitive_calls:
            # interrupt 会令当前节点从头重进，所以真实工具调用必须全部位于它之后。
            response = interrupt(
                {
                    "type": "tool_approval",
                    "message": "以下敏感工具调用需要人工审批",
                    "calls": [dict(call) for call in sensitive_calls],
                    "allowed_actions": ["approve", "reject", "edit"],
                }
            )
            decisions = _normalize_approval_decisions(response, sensitive_calls)

        for call in state["pending_tool_calls"]:
            name = call["name"]
            ok = False
            dedup = False
            approval = "not_required"
            try:
                arguments = json.loads(call["arguments"] or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("工具参数必须是 JSON 对象")
                if name in deps.sensitive_tools:
                    decision = decisions[call["id"]]
                    approval = decision["action"]
                    if approval == "reject":
                        result = f"用户拒绝执行工具调用: {decision['reason']}"
                        tool_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": result,
                            }
                        )
                        events.append(
                            {
                                "node": "tools",
                                "iteration": state["iteration"],
                                "tool": name,
                                "ok": False,
                                "dedup": False,
                                "approval": approval,
                                "arguments": arguments,
                                "detail": result[:120],
                            }
                        )
                        continue
                    if approval == "edit":
                        arguments = decision["arguments"]
                key = tool_key(name, arguments) if name in deps.idempotent_tools else None
                if key is not None and key in cache:
                    result = cache[key]
                    ok = True
                    dedup = True
                else:
                    result = deps.tool_runner(name, arguments)
                    ok = True
                    if key is not None:
                        cache[key] = result
            except Exception as exc:
                # 将可恢复错误反馈给模型，由下一轮决定换参数还是解释失败。
                result = f"工具执行出错: {exc}"

            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result,
                }
            )
            events.append(
                {
                    "node": "tools",
                    "iteration": state["iteration"],
                    "tool": name,
                    "ok": ok,
                    "dedup": dedup,
                    "approval": approval,
                    "arguments": arguments,
                    "detail": result[:120],
                }
            )

        return {
            "messages": tool_messages,
            "pending_tool_calls": [],
            "tool_cache": cache,
            "trace": events,
        }

    def force_stop_node(state: AgentState) -> dict:
        fallback = f"达到最大迭代次数({state['max_iterations']})仍未结束,已强制停止。"
        return {
            "messages": [{"role": "assistant", "content": fallback}],
            "answer": fallback,
            "stop_reason": "max_iterations",
            "trace": [
                {
                    "node": "force_stop",
                    "iteration": state["iteration"],
                    "detail": fallback,
                }
            ],
        }

    def route_after_model(state: AgentState) -> Literal["tools", "end"]:
        return "tools" if state["pending_tool_calls"] else "end"

    def route_after_tools(state: AgentState) -> Literal["model", "force_stop"]:
        if state["iteration"] >= state["max_iterations"]:
            return "force_stop"
        return "model"

    def observed(name: str, function: Callable[[AgentState], dict]):
        return observe_node(observer, f"{node_prefix}{name}", function)

    builder = StateGraph(AgentState)
    builder.add_node("model", observed("model", model_node))
    builder.add_node("tools", observed("tools", tool_node))
    builder.add_node("force_stop", observed("force_stop", force_stop_node))
    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model",
        route_after_model,
        {"tools": "tools", "end": END},
    )
    builder.add_conditional_edges(
        "tools",
        route_after_tools,
        {"model": "model", "force_stop": "force_stop"},
    )
    builder.add_edge("force_stop", END)
    return builder.compile(checkpointer=checkpointer)


def initial_agent_state(user_input: str, max_iterations: int = 10) -> AgentState:
    """校验参数并创建一条全新的单 Agent 状态。"""
    if not user_input.strip():
        raise ValueError("user_input 不能为空")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations 必须大于等于 1")
    return {
        "messages": [{"role": "user", "content": user_input}],
        "pending_tool_calls": [],
        "tool_cache": {},
        "iteration": 0,
        "max_iterations": max_iterations,
        "answer": "",
        "stop_reason": "",
        "trace": [],
    }


def invoke_agent(
    user_input: str,
    max_iterations: int = 10,
    graph=None,
) -> AgentState:
    """运行状态图并返回完整状态，供调试、评测和 API 层使用。"""
    app = graph or DEFAULT_GRAPH
    initial_state = initial_agent_state(user_input, max_iterations)
    # 一次模型调用最多对应一次工具节点；显式放宽图递归上限以匹配参数。
    return app.invoke(
        initial_state,
        {"recursion_limit": max_iterations * 2 + 5},
    )


def run_langgraph_agent(user_input: str, max_iterations: int = 10) -> str:
    """与手写版 ``run_agent`` 对齐的简洁文本接口。"""
    return invoke_agent(user_input, max_iterations=max_iterations)["answer"]


DEFAULT_GRAPH = build_agent_graph()
