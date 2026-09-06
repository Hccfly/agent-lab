"""LangGraph Planner -> scheduler -> Send 并行 Executor -> 汇总。

本阶段聚焦职责拆分与子图组合：
- Planner 节点只负责把目标拆成带角色的结构化任务；
- Executor 是独立子图，按角色调用基础单 Agent 图；
- Summarizer 节点只基于真实任务结果生成最终回答。

每个 superstep 执行一波就绪任务，Reducer 合并增量结果后进入下一波。
失败分支保留错误，依赖它的任务标记 blocked；独立分支仍继续。
"""
from __future__ import annotations

import operator
import json
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

import agent
from langgraph_agent import AgentDependencies, build_agent_graph, invoke_agent
from multiagent import (
    EXECUTOR_TEMPLATE,
    RETRY_TEMPLATE,
    ROLE_REGISTRY,
    _extract_json,
    _is_executor_failure,
    _strip_code_fence,
)
from observability import RunObserver, observe_node
from tools import IDEMPOTENT_TOOLS, run_tool


SUMMARIZER_SYSTEM = """你是多 Agent 系统的结果汇总者。
你只能根据用户原始目标和已完成子任务的真实结果组织最终回答，不得编造未出现的事实。
如果某个子任务失败，明确说明失败项及其影响。回答使用简体中文，简洁、完整。"""

# 单次完整规划：父图不循环重新规划，必须显式声明本批内的所有数据依赖。
DAG_PLANNER_SYSTEM = """你是任务规划者，只返回 JSON：
{"tasks": [{"task_id": 1, "agent": "calculator", "description": "...", "depends_on": []}]}。
一次列出完成目标需要的全部任务，task_id 为唯一正整数，depends_on 为本批前置任务 ID 数组。
无需工具即可回答时可直接返回文本。需要先获取数据再处理时，把处理任务依赖到获取任务；
不得编造尚未获取的数据。同层无依赖任务可以并行，任务列表不要求按依赖排序。
角色只能取 general/calculator/time/file。避免无意义拆分，不输出循环依赖。"""


def merge_task_maps(left: dict, right: dict) -> dict:
    """分支只返回自己的增量；相同 ID 的冲突不能悄悄覆盖。"""
    merged = dict(left)
    for key, value in right.items():
        if key in merged and merged[key] != value:
            raise ValueError(f"任务 {key} 的并行结果冲突")
        merged[key] = value
    return merged


class PlannedTask(TypedDict, total=False):
    task_id: int
    agent: str
    description: str
    reason: str
    depends_on: list[int]


class MultiTraceEvent(TypedDict, total=False):
    node: str
    task_id: int
    role: str
    tools: list[str]
    task_count: int
    retried: bool
    agent_path: list[str]
    detail: str
    wave: int
    task_ids: list[int]
    status: str


class MultiAgentState(TypedDict):
    """父图状态：只保存跨 Planner/Executor/Summarizer 共享的数据。"""

    goal: str
    tasks: list[PlannedTask]
    ready_tasks: list[PlannedTask]
    wave: int
    task_results: Annotated[dict[int, str], merge_task_maps]
    task_errors: Annotated[dict[int, str], merge_task_maps]
    task_status: Annotated[dict[int, str], merge_task_maps]
    planner_text: str
    answer: str
    stop_reason: str
    max_tasks: int
    trace: Annotated[list[MultiTraceEvent], operator.add]


class ExecutorState(TypedDict):
    """Executor 子图私有状态，与父图通过适配节点显式传递。"""

    task: PlannedTask
    prior_results: dict[int, str]
    role: str
    prompt: str
    result: str
    retried: bool
    agent_trace: list[dict[str, Any]]
    failed: bool
    trace: Annotated[list[MultiTraceEvent], operator.add]


@dataclass(frozen=True)
class MultiAgentDependencies:
    """父图和所有角色子图共享的可替换外部依赖。"""

    model_client: Any
    model: str
    tool_runner: Callable[[str, dict], str]
    role_registry: dict[str, dict[str, Any]]
    idempotent_tools: frozenset[str]
    executor_max_iterations: int = 6
    request_timeout_seconds: float = 30.0


def default_multiagent_dependencies() -> MultiAgentDependencies:
    return MultiAgentDependencies(
        model_client=agent.client,
        model=agent.MODEL,
        tool_runner=run_tool,
        role_registry=ROLE_REGISTRY,
        idempotent_tools=IDEMPOTENT_TOOLS,
        request_timeout_seconds=agent.MODEL_TIMEOUT_SECONDS,
    )


def _tool_names(tool_schemas: list[dict]) -> list[str]:
    return [schema["function"]["name"] for schema in tool_schemas]


def _normalize_tasks(
    raw_tasks: Any,
    max_tasks: int,
    role_registry: dict[str, dict[str, Any]],
) -> list[PlannedTask]:
    """严格校验任务 ID 和依赖；不能改写 ID 或截断列表而破坏引用。"""
    if not isinstance(raw_tasks, list):
        raise ValueError("tasks 必须是数组")
    if len(raw_tasks) > max_tasks:
        raise ValueError(f"任务数超过上限 {max_tasks}")

    tasks: list[PlannedTask] = []
    seen_ids: set[int] = set()
    for raw in raw_tasks:
        if not isinstance(raw, dict):
            raise ValueError("每个任务必须是对象")
        description = raw.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("任务 description 必须是非空字符串")
        task_id = raw.get("task_id")
        if type(task_id) is not int or task_id < 1 or task_id in seen_ids:
            raise ValueError(f"task_id 必须是唯一正整数: {task_id!r}")
        seen_ids.add(task_id)

        requested_role = str(raw.get("agent") or "general")
        role = requested_role if requested_role in role_registry else "general"
        task: PlannedTask = {
            "task_id": task_id,
            "agent": role,
            "description": description,
            "reason": str(raw.get("reason") or ""),
        }
        depends_on = raw.get("depends_on", [])
        if not isinstance(depends_on, list) or any(
            type(dep) is not int or dep < 1 for dep in depends_on
        ):
            raise ValueError(f"任务 {task_id} 的 depends_on 必须是正整数数组")
        task["depends_on"] = list(dict.fromkeys(depends_on))
        tasks.append(task)

    for task in tasks:
        missing = set(task["depends_on"]) - seen_ids
        if missing:
            raise ValueError(f"任务 {task['task_id']} 引用了不存在的任务 {sorted(missing)}")
    remaining = {t["task_id"]: set(t["depends_on"]) for t in tasks}
    done: set[int] = set()
    while remaining:
        ready = {tid for tid, dependencies in remaining.items() if dependencies <= done}
        if not ready:
            raise ValueError(f"依赖环涉及任务（含受阻后继）: {sorted(remaining)}")
        done.update(ready)
        remaining = {tid: dependencies for tid, dependencies in remaining.items() if tid not in ready}
    return tasks


def _executor_prompt(task: PlannedTask, prior_results: dict[int, str]) -> str:
    prompt = EXECUTOR_TEMPLATE.format(description=task["description"])
    if not prior_results:
        return prompt
    context = "\n".join(
        f"任务 {task_id}: {result}"
        for task_id, result in sorted(prior_results.items())
    )
    return (
        prompt
        + "\n\n以下是本次协作中已经完成的任务结果。需要时直接使用这些真实结果，"
        "不要假设或编造前置结论：\n"
        + context
    )


def build_executor_subgraph(
    dependencies: MultiAgentDependencies | None = None,
    observer: RunObserver | None = None,
):
    """构建 Executor 子图，并为每种角色预编译最小工具集单 Agent 图。"""
    deps = dependencies or default_multiagent_dependencies()
    role_graphs = {
        role: build_agent_graph(
            AgentDependencies(
                model_client=deps.model_client,
                model=deps.model,
                tool_schemas=cfg["tools"],
                tool_runner=deps.tool_runner,
                system_prompt=cfg["system"],
                idempotent_tools=deps.idempotent_tools,
                request_timeout_seconds=deps.request_timeout_seconds,
            ),
            observer=observer,
            node_prefix=f"role.{role}.",
        )
        for role, cfg in deps.role_registry.items()
    }

    def prepare_node(state: ExecutorState) -> dict:
        requested_role = state["task"].get("agent") or "general"
        role = requested_role if requested_role in deps.role_registry else "general"
        tools = _tool_names(deps.role_registry[role]["tools"])
        return {
            "role": role,
            "prompt": _executor_prompt(state["task"], state["prior_results"]),
            "trace": [
                {
                    "node": "executor_prepare",
                    "task_id": state["task"]["task_id"],
                    "role": role,
                    "tools": tools,
                }
            ],
        }

    def run_role_node(state: ExecutorState) -> dict:
        role = state["role"]
        agent_state = call_role(role, state["prompt"])
        agent_path = [event["node"] for event in agent_state["trace"]]
        return {
            "result": agent_state["answer"],
            "failed": agent_state["failed"],
            "agent_trace": agent_state["trace"],
            "trace": [
                {
                    "node": "executor_run",
                    "task_id": state["task"]["task_id"],
                    "role": role,
                    "tools": _tool_names(deps.role_registry[role]["tools"]),
                    "retried": False,
                    "agent_path": agent_path,
                    "detail": agent_state["answer"][:160],
                    "status": "failed" if agent_state["failed"] else "completed",
                }
            ],
        }

    def route_after_role(state: ExecutorState) -> Literal["retry_general", "end"]:
        return "retry_general" if state["failed"] else "end"

    def call_role(role: str, prompt: str) -> dict:
        try:
            output = invoke_agent(
                prompt, max_iterations=deps.executor_max_iterations,
                graph=role_graphs[role],
            )
        except Exception as exc:
            return {"answer": f"{type(exc).__name__}: {exc}", "trace": [], "failed": True}
        output["failed"] = (
            output["stop_reason"] != "completed"
            or not output["answer"].strip()
            or _is_executor_failure(output["answer"])
        )
        return output

    def retry_general_node(state: ExecutorState) -> dict:
        context = "\n".join(
            f"任务 {task_id}: {result}"
            for task_id, result in sorted(state["prior_results"].items())
        )
        retry_prompt = (
            RETRY_TEMPLATE
            .replace("{description}", state["task"]["description"])
            .replace("{failure}", state["result"][:600])
            .replace("{context}", context or "(本轮暂无其他已完成结果)")
        )
        agent_state = call_role("general", retry_prompt)
        return {
            "role": "general",
            "result": agent_state["answer"],
            "failed": agent_state["failed"],
            "retried": True,
            "agent_trace": agent_state["trace"],
            "trace": [
                {
                    "node": "executor_retry_general",
                    "task_id": state["task"]["task_id"],
                    "role": "general",
                    "tools": _tool_names(deps.role_registry["general"]["tools"]),
                    "retried": True,
                    "agent_path": [event["node"] for event in agent_state["trace"]],
                    "detail": agent_state["answer"][:160],
                    "status": "failed" if agent_state["failed"] else "completed",
                }
            ],
        }

    builder = StateGraph(ExecutorState)
    builder.add_node(
        "prepare", observe_node(observer, "executor.prepare", prepare_node)
    )
    builder.add_node(
        "run_role", observe_node(observer, "executor.run_role", run_role_node)
    )
    builder.add_node(
        "retry_general",
        observe_node(observer, "executor.retry_general", retry_general_node),
    )
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "run_role")
    builder.add_conditional_edges(
        "run_role",
        route_after_role,
        {"retry_general": "retry_general", "end": END},
    )
    builder.add_edge("retry_general", END)
    return builder.compile()


def build_multiagent_graph(
    dependencies: MultiAgentDependencies | None = None,
    checkpointer=None,
    interrupt_after=None,
    observer: RunObserver | None = None,
):
    """构建 Planner/Executor/Summarizer 父图。

    checkpointer 由调用方注入，便于生产使用 SQLite、测试使用内存或临时库。
    interrupt_after 仅用于断点恢复演示；常规执行不设置静态断点。
    """
    deps = dependencies or default_multiagent_dependencies()
    executor_subgraph = build_executor_subgraph(deps, observer=observer)

    def planner_node(state: MultiAgentState) -> dict:
        response = deps.model_client.chat.completions.create(
            model=deps.model,
            messages=[
                {"role": "system", "content": DAG_PLANNER_SYSTEM},
                {"role": "user", "content": f"目标: {state['goal']}\n最多 {state['max_tasks']} 个任务。"},
            ],
            timeout=deps.request_timeout_seconds,
        )
        planner_text = (response.choices[0].message.content or "").strip()
        stripped = _strip_code_fence(planner_text).strip()
        event: MultiTraceEvent = {"node": "planner", "detail": planner_text[:160]}

        if stripped and not stripped.startswith("{"):
            return {
                "planner_text": planner_text,
                "answer": planner_text,
                "stop_reason": "planner_direct",
                "trace": [event],
            }

        try:
            plan = _extract_json(planner_text)
            if not isinstance(plan, dict) or "tasks" not in plan:
                raise ValueError("规划必须包含 tasks")
            tasks = _normalize_tasks(
                plan.get("tasks", []), state["max_tasks"], deps.role_registry
            )
        except (ValueError, TypeError, AttributeError) as exc:
            event["detail"] = f"无效规划: {exc}"
            return {"planner_text": planner_text, "answer": event["detail"],
                    "stop_reason": "invalid_plan", "trace": [event]}

        event["task_count"] = len(tasks)
        if not tasks:
            return {
                "planner_text": planner_text,
                "answer": "Planner 未拆解出子任务，没有可执行内容。",
                "stop_reason": "no_tasks",
                "trace": [event],
            }
        return {
            "planner_text": planner_text,
            "tasks": tasks,
            "trace": [event],
        }

    def scheduler_node(state: MultiAgentState) -> dict:
        """上一波合流后才运行：传播失败，再计算所有就绪任务。"""
        status = dict(state["task_status"])
        blocked: dict[int, str] = {}
        while True:
            added = False
            for task in state["tasks"]:
                tid = task["task_id"]
                if tid in status:
                    continue
                failed_deps = [d for d in task["depends_on"] if status.get(d) in {"failed", "blocked"}]
                if failed_deps:
                    status[tid] = "blocked"
                    blocked[tid] = f"前置任务失败或受阻: {failed_deps}"
                    added = True
            if not added:
                break
        ready = [t for t in state["tasks"] if t["task_id"] not in status
                 and all(status.get(d) == "completed" for d in t["depends_on"])]
        events = [{"node": "scheduler", "task_id": tid, "status": "blocked", "detail": error}
                  for tid, error in sorted(blocked.items())]
        wave = state["wave"] + bool(ready)
        if ready:
            events.append({"node": "scheduler", "wave": wave,
                           "task_ids": [t["task_id"] for t in ready]})
        return {"ready_tasks": ready, "wave": wave,
                "task_status": {tid: "blocked" for tid in blocked},
                "task_errors": blocked, "trace": events}

    def dispatch_ready(state: MultiAgentState):
        if not state["ready_tasks"]:
            return "summarizer"
        # Send 独享输入：不把无关任务结果或父图整个状态交给分支。
        return [Send("executor", {"task": task, "wave": state["wave"],
                    "prior_results": {tid: state["task_results"][tid] for tid in task["depends_on"]}})
                for task in state["ready_tasks"]]

    def executor_node(state: dict) -> dict:
        """每个 Send 只处理一个任务，只返回该 task_id 的增量结果。"""
        task = state["task"]
        child_state: ExecutorState = {
            "task": task,
            "prior_results": dict(state["prior_results"]),
            "role": "",
            "prompt": "",
            "result": "",
            "retried": False,
            "agent_trace": [],
            "failed": False,
            "trace": [],
        }
        child_result = executor_subgraph.invoke(child_state)
        tid = task["task_id"]
        failed = child_result["failed"]
        return {
            "task_results": {} if failed else {tid: child_result["result"]},
            "task_errors": {tid: child_result["result"]} if failed else {},
            "task_status": {tid: "failed" if failed else "completed"},
            "trace": [{**event, "wave": state["wave"]} for event in child_result["trace"]],
        }

    def summarizer_node(state: MultiAgentState) -> dict:
        results_text = json.dumps([
            {"task_id": task["task_id"], "description": task["description"],
             "status": state["task_status"][task["task_id"]],
             "result": state["task_results"].get(task["task_id"]),
             "error": state["task_errors"].get(task["task_id"])}
            for task in sorted(state["tasks"], key=lambda t: t["task_id"])
        ], ensure_ascii=False)
        response = deps.model_client.chat.completions.create(
            model=deps.model,
            messages=[
                {"role": "system", "content": SUMMARIZER_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"用户原始目标:\n{state['goal']}\n\n"
                        f"已完成子任务结果:\n{results_text}\n\n"
                        "请生成最终回答。"
                    ),
                },
            ],
            timeout=deps.request_timeout_seconds,
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            answer = results_text or "没有可汇总的任务结果。"
        return {
            "answer": answer,
            "stop_reason": "partial_failure" if state["task_errors"] else "completed",
            "trace": [
                {
                    "node": "summarizer",
                    "task_count": len(state["task_results"]),
                    "detail": answer[:160],
                }
            ],
        }

    def route_after_planner(state: MultiAgentState) -> Literal["scheduler", "end"]:
        return "end" if state["stop_reason"] else "scheduler"

    builder = StateGraph(MultiAgentState)
    builder.add_node("planner", observe_node(observer, "planner", planner_node))
    builder.add_node(
        "scheduler", observe_node(observer, "scheduler", scheduler_node)
    )
    builder.add_node("executor", observe_node(observer, "executor", executor_node))
    builder.add_node(
        "summarizer", observe_node(observer, "summarizer", summarizer_node)
    )
    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {"scheduler": "scheduler", "end": END},
    )
    builder.add_conditional_edges("scheduler", dispatch_ready, ["executor", "summarizer"])
    # 整波 Send 节点完成后，LangGraph 合并所有增量，再唤醒一次 scheduler。
    builder.add_edge("executor", "scheduler")
    builder.add_edge("summarizer", END)
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after,
    )


def initial_multiagent_state(goal: str, max_tasks: int = 5) -> MultiAgentState:
    """校验运行参数并创建新的父图状态。"""
    if not goal.strip():
        raise ValueError("goal 不能为空")
    if type(max_tasks) is not int or max_tasks < 1:
        raise ValueError("max_tasks 必须大于等于 1")
    return {
        "goal": goal,
        "tasks": [],
        "ready_tasks": [],
        "wave": 0,
        "task_results": {},
        "task_errors": {},
        "task_status": {},
        "planner_text": "",
        "answer": "",
        "stop_reason": "",
        "max_tasks": max_tasks,
        "trace": [],
    }


def invoke_multiagent(
    goal: str,
    max_tasks: int = 5,
    graph=None,
    max_concurrency: int = 4,
) -> MultiAgentState:
    """运行多 Agent 父图并返回完整状态。"""
    if type(max_concurrency) is not int or max_concurrency < 1:
        raise ValueError("max_concurrency 必须是正整数")
    initial_state = initial_multiagent_state(goal, max_tasks)
    app = graph or DEFAULT_MULTIAGENT_GRAPH
    return app.invoke(initial_state, {"recursion_limit": 2 * max_tasks + 10,
                                      "max_concurrency": max_concurrency})


def run_langgraph_multiagent(goal: str, max_tasks: int = 5, max_concurrency: int = 4) -> str:
    return invoke_multiagent(goal, max_tasks=max_tasks, max_concurrency=max_concurrency)["answer"]


DEFAULT_MULTIAGENT_GRAPH = build_multiagent_graph()
