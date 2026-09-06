"""状态与记忆层:会话持久化 + 会话管理。

设计要点:
- 状态(state): 对话历史写进磁盘(sessions/<id>.json),agent 重启后
  用同一个 session_id 能接着上次继续聊——这就是"agent 是有状态的"。
- 每个会话存两份东西:
    summary  -> 长期记忆:早期对话的摘要(由 agent 调用模型生成)
    messages -> 短期记忆:最近的原始消息(完整保留了 tool 调用配对)
"""
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from governance import atomic_write_json, locked_path, validate_identifier

SESSION_DIR = Path(__file__).parent / "sessions"


def _path(session_id: str) -> Path:
    return SESSION_DIR / f"{validate_session_id(session_id)}.json"


def validate_session_id(session_id: str) -> str:
    return validate_identifier(session_id, "session_id")


@contextmanager
def session_lock(session_id: str) -> Iterator[None]:
    """同一 session 串行化完整读改写事务，不同 session 互不阻塞。"""
    with locked_path(_path(session_id)):
        yield


def load_session(session_id: str) -> dict:
    """读取会话,返回 {"summary": str, "messages": [...]};不存在则返回空会话。"""
    path = _path(session_id)
    with locked_path(path):
        if not path.exists():
            return {"summary": "", "messages": []}
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    data.setdefault("summary", "")
    data.setdefault("messages", [])
    return data


def save_session(session_id: str, summary: str, messages: list) -> None:
    atomic_write_json(
        _path(session_id),
        {"summary": summary, "messages": messages},
        indent=2,
    )


def list_sessions() -> list[tuple[str, int]]:
    """列出所有会话及其消息数,新的在前。"""
    if not SESSION_DIR.exists():
        return []
    result = []
    for p in SESSION_DIR.glob("*.json"):
        try:
            with locked_path(p):
                data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        result.append((p.stem, len(data.get("messages", []))))
    result.sort(reverse=True)
    return result
