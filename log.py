"""结构化日志:JSON Lines 写入文件 + 终端人类可读。

设计要点:
- 一条日志两个输出:终端(stderr,人类可读)+ 文件 logs/agent.jsonl(JSON Lines)。
- JSON Lines:每行一个合法 JSON 对象,可被 ELK / 日志平台直接消费,
  是生产级 agent 的标准做法,让日志可检索、可聚合、可审计。
- 结构化字段:event(事件类型)、session_id(跨调用关联)、tool、
  latency_ms(工具耗时)、tokens 等,支撑监控与排障。
"""
import json
import logging
import sys
from datetime import datetime

_LOG_FILE = "logs/agent.jsonl"

_logger = None


class _JsonLineFormatter(logging.Formatter):
    """把日志记录序列化为单行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": getattr(record, "event", "log"),
        }
        fields = getattr(record, "fields", None)
        if fields:
            data.update(fields)
        return json.dumps(data, ensure_ascii=False)


def setup_logging(log_file: str = _LOG_FILE, console: bool = True) -> logging.Logger:
    """幂等初始化日志器。文件输出 JSON Lines,终端输出人类可读。"""
    global _logger
    if _logger is not None:
        return _logger

    from pathlib import Path

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("agent")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 避免重复输出到 root logger

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(_JsonLineFormatter())
    logger.addHandler(fh)

    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)

    _logger = logger
    return logger


def log_event(event: str, msg: str = "", level: int = logging.INFO, **fields) -> None:
    """发一条结构化日志。msg 是人类可读文本(进终端),fields 进 JSON。"""
    logger = setup_logging()
    logger.log(level, msg, extra={"event": event, "fields": fields})
