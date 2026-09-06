"""运行时可观测层：节点跨度、状态增量、失败位置与轨迹导出。

观察器只包裹节点函数，不写入 AgentState，因此不会改变 checkpoint schema，
也不会把锁、回调等不可序列化对象带进 LangGraph 状态。
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Literal

from governance import atomic_write_json, atomic_write_text


TraceCallback = Callable[[dict[str, Any]], None]
RunMode = Literal["single", "multi"]


def serialize_error(exc: BaseException) -> dict[str, str]:
    """生成适合日志和 SSE 的有限错误信息，不包含本地堆栈与环境变量。"""
    message = str(exc).strip() or type(exc).__name__
    return {"type": type(exc).__name__, "message": message[:500]}


def _preview(text: str) -> dict[str, Any]:
    normalized = text.replace("\r", " ").replace("\n", " ").strip()
    return {"chars": len(text), "preview": normalized[:120]}


def _summarize_value(key: str, value: Any) -> Any:
    """压缩 State，保留调试所需变化，不复制完整对话和工具结果。"""
    if key == "messages" and isinstance(value, list):
        return {
            "count": len(value),
            "roles": [
                item.get("role", "unknown")
                for item in value
                if isinstance(item, dict)
            ],
        }
    if key == "pending_tool_calls" and isinstance(value, list):
        return [
            {"id": item.get("id"), "name": item.get("name")}
            for item in value
            if isinstance(item, dict)
        ]
    if key == "tool_cache" and isinstance(value, dict):
        return {
            "entries": len(value),
            "keys": [str(item)[:80] for item in list(value)[:8]],
        }
    if key in {"tasks", "ready_tasks"} and isinstance(value, list):
        return [
            {
                "task_id": item.get("task_id"),
                "agent": item.get("agent"),
                "depends_on": item.get("depends_on", []),
            }
            for item in value
            if isinstance(item, dict)
        ]
    if key == "task" and isinstance(value, dict):
        return {
            "task_id": value.get("task_id"),
            "agent": value.get("agent"),
            "depends_on": value.get("depends_on", []),
        }
    if key in {"task_results", "task_errors"} and isinstance(value, dict):
        return {
            "task_ids": sorted(value, key=lambda item: str(item)),
            "count": len(value),
        }
    if key == "task_status" and isinstance(value, dict):
        return {
            str(item): status
            for item, status in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if key == "trace" and isinstance(value, list):
        return {
            "count": len(value),
            "nodes": [
                item.get("node") for item in value if isinstance(item, dict)
            ],
        }
    if key in {"answer", "planner_text", "result", "detail"} and isinstance(value, str):
        return _preview(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value if not isinstance(value, str) else value[:160]
    if isinstance(value, list):
        return {"count": len(value)}
    if isinstance(value, dict):
        return {"keys": sorted(str(item) for item in value)[:20]}
    return {"type": type(value).__name__}


def summarize_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {"type": type(state).__name__}
    return {
        str(key): _summarize_value(str(key), value)
        for key, value in state.items()
    }


class RunObserver:
    """线程安全的单次运行观察器，支持 LangGraph Send 并行分支。"""

    def __init__(
        self,
        *,
        run_id: str | None = None,
        on_event: TraceCallback | None = None,
    ) -> None:
        self.run_id = run_id or uuid.uuid4().hex
        self._on_event = on_event
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._sequence = 0
        self._started_at = time.perf_counter()
        self._started = False
        self._finished = False

    @property
    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._events)

    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._sequence += 1
            item = {
                "schema_version": 1,
                "run_id": self.run_id,
                "seq": self._sequence,
                "timestamp": now.isoformat(timespec="milliseconds"),
                "elapsed_ms": round(
                    (time.perf_counter() - self._started_at) * 1000, 3
                ),
                "event": event,
                **payload,
            }
            self._events.append(item)
        if self._on_event is not None:
            try:
                self._on_event(deepcopy(item))
            except Exception:
                # 观测链路故障不能改变 Agent 的业务执行结果。
                pass
        return item

    def start(self, *, mode: RunMode, input_text: str) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        self.emit("run_start", mode=mode, input=_preview(input_text))

    def node_start(self, node: str, state: Any) -> tuple[str, float, float]:
        span_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        event = self.emit(
            "node_start",
            span_id=span_id,
            node=node,
            state=summarize_state(state),
        )
        return span_id, started, float(event["elapsed_ms"])

    def node_end(
        self,
        node: str,
        span_id: str,
        started: float,
        started_ms: float,
        delta: Any,
    ) -> None:
        self.emit(
            "node_end",
            span_id=span_id,
            node=node,
            status="completed",
            started_ms=started_ms,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
            state_delta=summarize_state(delta),
        )

    def node_error(
        self,
        node: str,
        span_id: str,
        started: float,
        started_ms: float,
        exc: BaseException,
    ) -> None:
        self.emit(
            "node_error",
            span_id=span_id,
            node=node,
            status="failed",
            started_ms=started_ms,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
            error=serialize_error(exc),
        )

    def finish(
        self,
        status: Literal["completed", "failed"],
        *,
        result: dict[str, Any] | None = None,
        error: dict[str, str] | None = None,
    ) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
        payload: dict[str, Any] = {"status": status}
        if result is not None:
            payload["result"] = summarize_state(result)
        if error is not None:
            payload["error"] = error
        self.emit("run_end", **payload)


def observe_node(
    observer: RunObserver | None,
    node: str,
    function: Callable[[Any], dict[str, Any]],
) -> Callable[[Any], dict[str, Any]]:
    """为同步节点增加 span；observer=None 时保持原始调用，无运行期开销。"""
    if observer is None:
        return function

    @wraps(function)
    def wrapped(state: Any) -> dict[str, Any]:
        span_id, started, started_ms = observer.node_start(node, state)
        try:
            delta = function(state)
        except Exception as exc:
            observer.node_error(node, span_id, started, started_ms, exc)
            raise
        observer.node_end(node, span_id, started, started_ms, delta)
        return delta

    return wrapped


def run_observed_graph(
    input_text: str,
    *,
    mode: RunMode = "single",
    observer: RunObserver | None = None,
    dependencies: Any = None,
    max_iterations: int = 10,
    max_tasks: int = 5,
    max_concurrency: int = 4,
) -> tuple[dict[str, Any], RunObserver]:
    """构建带观察器的图并运行；依赖可替换，离线测试不请求真实 API。"""
    active = observer or RunObserver()
    active.start(mode=mode, input_text=input_text)
    try:
        if mode == "single":
            from langgraph_agent import build_agent_graph, invoke_agent

            graph = build_agent_graph(dependencies=dependencies, observer=active)
            result = invoke_agent(
                input_text,
                max_iterations=max_iterations,
                graph=graph,
            )
        elif mode == "multi":
            from langgraph_multiagent import build_multiagent_graph, invoke_multiagent

            graph = build_multiagent_graph(
                dependencies=dependencies,
                observer=active,
            )
            result = invoke_multiagent(
                input_text,
                max_tasks=max_tasks,
                graph=graph,
                max_concurrency=max_concurrency,
            )
        else:
            raise ValueError(f"不支持的运行模式: {mode}")
    except Exception as exc:
        active.finish("failed", error=serialize_error(exc))
        raise
    active.finish("completed", result=result)
    return result, active


def _markdown_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render_trace_markdown(
    events: list[dict[str, Any]],
    *,
    title: str = "Agent 运行轨迹",
) -> str:
    """将统一事件转成可放进 README/设计文档的 Mermaid + 时间表。"""
    spans = [
        event
        for event in events
        if event.get("event") in {"node_end", "node_error"}
    ]
    lines = [
        f"# {title}",
        "",
        "## 节点事件完成顺序（并行以时间表为准）",
        "",
        "```mermaid",
        "flowchart LR",
    ]
    if spans:
        node_ids = []
        for index, event in enumerate(spans, 1):
            node_id = f"n{index}"
            node_ids.append(node_id)
            node = str(event.get("node", "unknown")).replace('"', "'")
            duration = event.get("duration_ms", 0)
            status = "失败" if event.get("status") == "failed" else "完成"
            lines.append(
                f'    {node_id}["{node}<br/>{duration} ms · {status}"]'
            )
        for left, right in zip(node_ids, node_ids[1:]):
            lines.append(f"    {left} --> {right}")
        failed = [
            f"n{index}"
            for index, event in enumerate(spans, 1)
            if event.get("status") == "failed"
        ]
        if failed:
            lines.append(
                "    classDef failed fill:#fee2e2,stroke:#dc2626,color:#7f1d1d"
            )
            lines.append(f"    class {','.join(failed)} failed")
    else:
        lines.append('    empty["没有节点事件"]')
    lines.extend(
        [
            "```",
            "",
            "## 节点时间线",
            "",
            "| Seq | 节点 | 开始(ms) | 耗时(ms) | 状态 | 状态变化 / 错误 |",
            "|---:|---|---:|---:|---|---|",
        ]
    )
    for event in spans:
        detail = event.get("state_delta") or event.get("error") or ""
        lines.append(
            "| {seq} | `{node}` | {started} | {duration} | {status} | {detail} |".format(
                seq=event.get("seq", ""),
                node=_markdown_cell(event.get("node", "unknown")),
                started=event.get("started_ms", ""),
                duration=event.get("duration_ms", ""),
                status="失败" if event.get("status") == "failed" else "完成",
                detail=_markdown_cell(detail),
            )
        )
    return "\n".join(lines) + "\n"


def write_trace_report(
    output_base: str | Path,
    observer: RunObserver,
    *,
    metadata: dict[str, Any] | None = None,
    title: str = "Agent 运行轨迹",
) -> tuple[Path, Path]:
    """原子写出机器可读 JSON 与人可读 Markdown。"""
    base = Path(output_base)
    if base.suffix.lower() in {".json", ".md"}:
        base = base.with_suffix("")
    json_path = base.with_suffix(".json")
    markdown_path = base.with_suffix(".md")
    payload = {
        "schema_version": 1,
        "run_id": observer.run_id,
        "metadata": metadata or {},
        "events": observer.events,
    }
    atomic_write_json(json_path, payload, indent=2)
    atomic_write_text(
        markdown_path,
        render_trace_markdown(payload["events"], title=title),
    )
    return json_path, markdown_path
