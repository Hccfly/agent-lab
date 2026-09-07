"""共用治理原语：标识校验、路径白名单、原子写和执行限流。"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar


PROJECT_ROOT = Path(__file__).resolve().parent
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()
T = TypeVar("T")


class PathAccessDenied(ValueError):
    """请求路径不在允许读取的根目录内。"""


class ExecutionTimeout(TimeoutError):
    """受治理操作超过执行时间上限。"""


class RateLimitExceeded(RuntimeError):
    """并发槽位已满，调用在排队期限内未获准执行。"""


def env_positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是数字") from exc
    if value <= 0:
        raise ValueError(f"环境变量 {name} 必须大于 0")
    return value


def env_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是整数") from exc
    if value < 1:
        raise ValueError(f"环境变量 {name} 必须大于等于 1")
    return value


def validate_identifier(value: str, field_name: str = "identifier") -> str:
    """限制持久化键，避免路径穿越、空白别名和超长文件名。"""
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} 必须以字母或数字开头，只能包含字母、数字、点、"
            "下划线或短横线，长度为 1-128"
        )
    return value


def allowed_read_roots() -> tuple[Path, ...]:
    """读取白名单；默认仅项目目录，可用环境变量追加或替换。"""
    raw = os.getenv("AGENT_ALLOWED_READ_ROOTS")
    candidates = raw.split(os.pathsep) if raw else [str(PROJECT_ROOT)]
    roots = tuple(
        Path(item.strip()).expanduser().resolve()
        for item in candidates
        if item.strip()
    )
    if not roots:
        raise ValueError("AGENT_ALLOWED_READ_ROOTS 至少要包含一个目录")
    return roots


def resolve_allowed_read_path(path: str) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise PathAccessDenied("文件路径不能为空")
    requested = Path(path).expanduser()
    if not requested.is_absolute():
        requested = PROJECT_ROOT / requested
    resolved = requested.resolve()
    roots = allowed_read_roots()
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise PathAccessDenied(
            f"路径不在允许读取的目录中: {resolved}; allowed_roots="
            + ", ".join(str(root) for root in roots)
        )
    return resolved


def path_lock(path: str | Path) -> threading.RLock:
    key = str(Path(path).resolve())
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def locked_path(path: str | Path) -> Iterator[None]:
    with path_lock(path):
        yield


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """同目录临时文件 + fsync + os.replace，避免读到半写入文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    with locked_path(target):
        try:
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            temporary = Path(temporary_name)
            with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, value: Any, *, indent: int | None = None) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=indent),
    )


class ExecutionGovernor:
    """以独立线程执行操作，同时限制并发、排队时间和执行时间。"""

    def __init__(
        self,
        max_concurrency: int,
        timeout_seconds: float,
        queue_timeout_seconds: float,
    ) -> None:
        if type(max_concurrency) is not int or max_concurrency < 1:
            raise ValueError("max_concurrency 必须是正整数")
        if timeout_seconds <= 0 or queue_timeout_seconds <= 0:
            raise ValueError("timeout_seconds 和 queue_timeout_seconds 必须大于 0")
        self.max_concurrency = max_concurrency
        self.timeout_seconds = float(timeout_seconds)
        self.queue_timeout_seconds = float(queue_timeout_seconds)
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix="agent-governor",
        )

    def run(self, operation: str, function: Callable[[], T]) -> T:
        if not self._slots.acquire(timeout=self.queue_timeout_seconds):
            raise RateLimitExceeded(
                f"{operation} 并发已达上限 {self.max_concurrency}，请稍后重试"
            )
        try:
            future = self._executor.submit(function)
        except Exception:
            self._slots.release()
            raise
        release_here = True
        try:
            return future.result(timeout=self.timeout_seconds)
        except FutureTimeoutError as exc:
            # Python 无法安全终止正在运行的线程；槽位必须等真实工作结束才释放。
            release_here = False
            future.add_done_callback(lambda _future: self._slots.release())
            future.cancel()
            raise ExecutionTimeout(
                f"{operation} 超过 {self.timeout_seconds:g} 秒执行上限"
            ) from exc
        finally:
            if release_here:
                self._slots.release()

    def close(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)
