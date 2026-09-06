"""为 LangGraph 多 Agent 父图提供 SQLite checkpoint 与恢复入口。"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from langgraph.checkpoint.sqlite import SqliteSaver

from governance import validate_identifier

from langgraph_multiagent import (
    MultiAgentDependencies,
    MultiAgentState,
    build_multiagent_graph,
    initial_multiagent_state,
)


DEFAULT_CHECKPOINT_DB = Path("checkpoints/agent_state.sqlite3")


class CheckpointThreadError(ValueError):
    """checkpoint 线程的使用方式不合法。"""


@dataclass(frozen=True)
class CheckpointRun:
    """一次 start/resume 后可供 CLI 和测试检查的状态快照。"""

    thread_id: str
    state: MultiAgentState
    next_nodes: tuple[str, ...]
    checkpoint_id: str | None

    @property
    def completed(self) -> bool:
        return not self.next_nodes


def validate_thread_id(thread_id: str) -> str:
    try:
        return validate_identifier(thread_id, "thread_id")
    except ValueError as exc:
        raise CheckpointThreadError(str(exc)) from exc


class PersistentMultiAgent:
    """SQLite 持久化的多 Agent 运行器。

    一个实例持有一条 SQLite 连接；使用 with 或 close() 明确释放它。
    interrupt_after 主要服务于教学 Demo，例如 ["executor"] 表示每个执行波次后暂停。
    """

    def __init__(
        self,
        database_path: str | Path = DEFAULT_CHECKPOINT_DB,
        dependencies: MultiAgentDependencies | None = None,
        interrupt_after: Sequence[str] | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._checkpointer = SqliteSaver(self._connection)
        self.graph = build_multiagent_graph(
            dependencies,
            checkpointer=self._checkpointer,
            interrupt_after=list(interrupt_after) if interrupt_after else None,
        )
        self._closed = False

    @staticmethod
    def _config(thread_id: str, max_tasks: int, max_concurrency: int) -> dict:
        if type(max_concurrency) is not int or max_concurrency < 1:
            raise ValueError("max_concurrency 必须是正整数")
        return {
            "configurable": {"thread_id": validate_thread_id(thread_id)},
            "recursion_limit": 2 * max_tasks + 10,
            "max_concurrency": max_concurrency,
        }

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("PersistentMultiAgent 已关闭")

    def _snapshot(self, config: dict):
        self._ensure_open()
        return self.graph.get_state(config)

    @staticmethod
    def _has_checkpoint(snapshot) -> bool:
        return bool(snapshot.values)

    @staticmethod
    def _to_result(thread_id: str, snapshot) -> CheckpointRun:
        configurable = (snapshot.config or {}).get("configurable", {})
        return CheckpointRun(
            thread_id=thread_id,
            state=dict(snapshot.values),
            next_nodes=tuple(snapshot.next),
            checkpoint_id=configurable.get("checkpoint_id"),
        )

    def start(
        self,
        goal: str,
        thread_id: str,
        max_tasks: int = 5,
        max_concurrency: int = 4,
    ) -> CheckpointRun:
        """创建新线程；已有 thread_id 一律拒绝，避免覆盖或重复执行。"""
        initial_state = initial_multiagent_state(goal, max_tasks)
        config = self._config(thread_id, max_tasks, max_concurrency)
        snapshot = self._snapshot(config)
        if self._has_checkpoint(snapshot):
            raise CheckpointThreadError(
                f"thread_id {thread_id!r} 已存在；请调用 resume() 或换一个新 ID"
            )
        self.graph.invoke(initial_state, config)
        return self._to_result(validate_thread_id(thread_id), self._snapshot(config))

    def resume(
        self,
        thread_id: str,
        max_concurrency: int = 4,
    ) -> CheckpointRun:
        """从最近 checkpoint 继续；已完成线程幂等返回，不会重新执行节点。"""
        normalized = validate_thread_id(thread_id)
        probe_config = self._config(normalized, 1, max_concurrency)
        before = self._snapshot(probe_config)
        if not self._has_checkpoint(before):
            raise CheckpointThreadError(f"thread_id {normalized!r} 不存在")
        max_tasks = int(before.values.get("max_tasks", 1))
        config = self._config(normalized, max_tasks, max_concurrency)
        if not before.next:
            return self._to_result(normalized, before)
        self.graph.invoke(None, config)
        return self._to_result(normalized, self._snapshot(config))

    def status(self, thread_id: str) -> CheckpointRun:
        """只读获取线程最新状态。"""
        normalized = validate_thread_id(thread_id)
        config = self._config(normalized, 1, 1)
        snapshot = self._snapshot(config)
        if not self._has_checkpoint(snapshot):
            raise CheckpointThreadError(f"thread_id {normalized!r} 不存在")
        return self._to_result(normalized, snapshot)

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> PersistentMultiAgent:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
