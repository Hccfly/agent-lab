"""基于 LangGraph interrupt 的敏感工具人工审批。"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from checkpointing import CheckpointThreadError, validate_thread_id
from langgraph_agent import (
    AgentDependencies,
    AgentState,
    build_agent_graph,
    default_dependencies,
    initial_agent_state,
)


DEFAULT_HITL_DB = Path("checkpoints/hitl_state.sqlite3")


class ApprovalThreadError(CheckpointThreadError):
    """HITL 线程不存在、冲突或当前并未等待审批。"""


@dataclass(frozen=True)
class ApprovalRun:
    thread_id: str
    state: AgentState
    next_nodes: tuple[str, ...]
    pending_calls: tuple[dict[str, Any], ...]
    checkpoint_id: str | None

    @property
    def waiting_for_approval(self) -> bool:
        return bool(self.pending_calls)

    @property
    def completed(self) -> bool:
        return not self.next_nodes


class PersistentApprovalAgent:
    """使用 SQLite 保存 interrupt，并通过 Command(resume=...) 提交审批。"""

    def __init__(
        self,
        database_path: str | Path = DEFAULT_HITL_DB,
        dependencies: AgentDependencies | None = None,
        sensitive_tools: Sequence[str] = ("read_file",),
    ) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._checkpointer = SqliteSaver(self._connection)
        base_dependencies = dependencies or default_dependencies()
        self.dependencies = replace(
            base_dependencies,
            sensitive_tools=frozenset(sensitive_tools),
        )
        self.graph = build_agent_graph(
            self.dependencies,
            checkpointer=self._checkpointer,
        )
        self._closed = False

    @staticmethod
    def _config(thread_id: str, max_iterations: int) -> dict:
        return {
            "configurable": {"thread_id": validate_thread_id(thread_id)},
            "recursion_limit": max_iterations * 2 + 5,
        }

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("PersistentApprovalAgent 已关闭")

    def _snapshot(self, config: dict):
        self._ensure_open()
        return self.graph.get_state(config)

    @staticmethod
    def _pending_calls(snapshot) -> tuple[dict[str, Any], ...]:
        calls: list[dict[str, Any]] = []
        for task in snapshot.tasks:
            for pending_interrupt in task.interrupts:
                payload = pending_interrupt.value
                if isinstance(payload, dict) and payload.get("type") == "tool_approval":
                    calls.extend(dict(call) for call in payload.get("calls", []))
        return tuple(calls)

    @classmethod
    def _to_result(cls, thread_id: str, snapshot) -> ApprovalRun:
        configurable = (snapshot.config or {}).get("configurable", {})
        return ApprovalRun(
            thread_id=thread_id,
            state=dict(snapshot.values),
            next_nodes=tuple(snapshot.next),
            pending_calls=cls._pending_calls(snapshot),
            checkpoint_id=configurable.get("checkpoint_id"),
        )

    def start(
        self,
        user_input: str,
        thread_id: str,
        max_iterations: int = 10,
    ) -> ApprovalRun:
        normalized = validate_thread_id(thread_id)
        initial_state = initial_agent_state(user_input, max_iterations)
        config = self._config(normalized, max_iterations)
        before = self._snapshot(config)
        if before.values:
            raise ApprovalThreadError(
                f"thread_id {normalized!r} 已存在；请审批现有调用或换一个新 ID"
            )
        self.graph.invoke(initial_state, config)
        return self._to_result(normalized, self._snapshot(config))

    def status(self, thread_id: str) -> ApprovalRun:
        normalized = validate_thread_id(thread_id)
        config = self._config(normalized, 1)
        snapshot = self._snapshot(config)
        if not snapshot.values:
            raise ApprovalThreadError(f"thread_id {normalized!r} 不存在")
        return self._to_result(normalized, snapshot)

    def decide(
        self,
        thread_id: str,
        decisions: Sequence[dict[str, Any]],
    ) -> ApprovalRun:
        """原子提交当前 interrupt 中所有敏感调用的决策。"""
        before = self.status(thread_id)
        if not before.waiting_for_approval:
            raise ApprovalThreadError(f"thread_id {before.thread_id!r} 当前没有待审批调用")
        max_iterations = int(before.state.get("max_iterations", 1))
        config = self._config(before.thread_id, max_iterations)
        self.graph.invoke(
            Command(resume={"decisions": [dict(item) for item in decisions]}),
            config,
        )
        return self._to_result(before.thread_id, self._snapshot(config))

    def approve(self, thread_id: str) -> ApprovalRun:
        before = self.status(thread_id)
        return self.decide(
            thread_id,
            [
                {"call_id": call["id"], "action": "approve"}
                for call in before.pending_calls
            ],
        )

    def reject(self, thread_id: str, reason: str) -> ApprovalRun:
        if not reason.strip():
            raise ValueError("拒绝原因不能为空")
        before = self.status(thread_id)
        return self.decide(
            thread_id,
            [
                {
                    "call_id": call["id"],
                    "action": "reject",
                    "reason": reason.strip(),
                }
                for call in before.pending_calls
            ],
        )

    def edit(
        self,
        thread_id: str,
        arguments: dict[str, Any],
    ) -> ApprovalRun:
        before = self.status(thread_id)
        if len(before.pending_calls) != 1:
            raise ApprovalThreadError("edit 快捷入口要求当前恰好有一个待审批调用")
        return self.decide(
            thread_id,
            [
                {
                    "call_id": before.pending_calls[0]["id"],
                    "action": "edit",
                    "arguments": arguments,
                }
            ],
        )

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> PersistentApprovalAgent:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
