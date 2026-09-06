"""可观测运行入口：实时打印节点事件并导出 JSON/Markdown 轨迹。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

from langgraph_agent import AgentDependencies
from observability import RunObserver, run_observed_graph, write_trace_report
from tools import IDEMPOTENT_TOOLS, TOOLS


PROJECT_ROOT = Path(__file__).resolve().parent


class _DemoClient:
    """固定走 model -> tools -> model，供无 API Key 的回归与排障演示。"""

    def __init__(self) -> None:
        self._step = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **_kwargs):
        self._step += 1
        if self._step == 1:
            tool_call = SimpleNamespace(
                id="demo-call-1",
                function=SimpleNamespace(
                    name="calculator",
                    arguments='{"expression":"(12+8)*3"}',
                ),
            )
            message = SimpleNamespace(content="", tool_calls=[tool_call])
        else:
            message = SimpleNamespace(content="计算结果是 60。", tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=None,
        )


def _offline_demo_dependencies() -> AgentDependencies:
    def runner(name: str, arguments: dict) -> str:
        if name != "calculator" or arguments != {"expression": "(12+8)*3"}:
            raise ValueError("离线演示只接受固定 calculator 调用")
        return "60"

    return AgentDependencies(
        model_client=_DemoClient(),
        model="offline-demo",
        tool_schemas=TOOLS,
        tool_runner=runner,
        system_prompt="offline demo",
        idempotent_tools=IDEMPOTENT_TOOLS,
    )


def _print_event(event: dict) -> None:
    kind = event["event"]
    if kind == "node_start":
        print(f"[{event['elapsed_ms']:>8.3f} ms] START {event['node']}")
    elif kind == "node_end":
        print(
            f"[{event['elapsed_ms']:>8.3f} ms] END   {event['node']} "
            f"({event['duration_ms']:.3f} ms)"
        )
    elif kind == "node_error":
        error = event["error"]
        print(
            f"[{event['elapsed_ms']:>8.3f} ms] ERROR {event['node']} "
            f"({error['type']}: {error['message']})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 LangGraph Agent，并导出节点、状态增量、耗时和失败位置。"
    )
    parser.add_argument("input", help="单 Agent 问题或多 Agent 目标")
    parser.add_argument(
        "--mode",
        choices=("single", "multi"),
        default="single",
        help="运行单 Agent 或 DAG 多 Agent",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports" / "latest-trace"),
        help="输出基名；会同时生成 .json 和 .md",
    )
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-tasks", type=int, default=5)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument(
        "--offline-demo",
        action="store_true",
        help="使用固定 FakeClient 演示，不请求真实 API（仅 single 模式）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    observer = RunObserver(on_event=_print_event)
    exit_code = 0
    answer = ""
    if args.offline_demo and args.mode != "single":
        print("--offline-demo 仅支持 --mode single", file=sys.stderr)
        return 2
    injected_dependencies = _offline_demo_dependencies() if args.offline_demo else None
    try:
        result, _ = run_observed_graph(
            args.input,
            mode=args.mode,
            observer=observer,
            dependencies=injected_dependencies,
            max_iterations=args.max_iterations,
            max_tasks=args.max_tasks,
            max_concurrency=args.max_concurrency,
        )
        answer = result.get("answer", "")
    except Exception as exc:
        exit_code = 1
        print(f"运行失败：{type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        json_path, markdown_path = write_trace_report(
            args.output,
            observer,
            metadata={"mode": args.mode, "input": args.input},
            title=f"Agent 运行轨迹（{args.mode}）",
        )
        print(f"JSON: {json_path.resolve()}")
        print(f"Markdown: {markdown_path.resolve()}")

    if answer:
        print(f"\n回答：\n{answer}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
