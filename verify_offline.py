"""离线验收总入口：轨迹、关键回归与已发布基准的一致性检查。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def _run(label: str, *arguments: str) -> None:
    print(f"\n{'=' * 12} {label} {'=' * 12}", flush=True)
    environment = os.environ.copy()
    environment["LANGCHAIN_TRACING_V2"] = "false"
    environment["LANGSMITH_TRACING"] = "false"
    completed = subprocess.run(
        [PYTHON, "-X", "utf8", *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{label} 失败，exit_code={completed.returncode}")


def _verify_baseline_report() -> dict:
    report_path = PROJECT_ROOT / "reports" / "offline-eval.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    recovery = payload["recovery"]
    total_runs = sum(version["runs"] for version in summary.values())

    if total_runs != 48:
        raise ValueError(f"基准样本数漂移：expected=48 actual={total_runs}")
    if any(version["expectation_pass_rate"] != 1.0 for version in summary.values()):
        raise ValueError("预期行为通过率不再是 100%")
    if recovery["v1"]["duplicate_executions"] != 2:
        raise ValueError("V1 恢复重复次数与基准报告不一致")
    if recovery["v2"]["duplicate_executions"] != 0:
        raise ValueError("V2 恢复出现重复执行")

    return {
        "total_runs": total_runs,
        "v1_tool_requests": summary["v1"]["tool_requests"],
        "v1_tool_executions": summary["v1"]["tool_executions"],
        "v2_tool_requests": summary["v2"]["tool_requests"],
        "v2_tool_executions": summary["v2"]["tool_executions"],
        "v1_replayed": recovery["v1"]["duplicate_executions"],
        "v2_replayed": recovery["v2"]["duplicate_executions"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="完全离线验证 Agent Lab，不请求真实模型 API。"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="只验证轨迹与可观测层；默认还验证恢复、审批和 SSE",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        _run(
            "阶段 1：节点轨迹",
            "trace_cli.py",
            "计算 (12+8)*3",
            "--offline-demo",
            "--output",
            "reports/offline-trace",
        )
        _run("阶段 2：可观测回归", "tests/test_observability.py")
        if not args.quick:
            _run("阶段 3：checkpoint 恢复", "tests/test_checkpoint_resume.py")
            _run("阶段 4：HITL 审批", "tests/test_hitl.py")
            _run("阶段 5：结构化 SSE", "tests/test_server.py")

        metrics = _verify_baseline_report()
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"\nVERIFICATION FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\n============ OFFLINE VERIFICATION PASSED ============")
    print("轨迹报告：reports/offline-trace.md / .json")
    print(
        f"离线基准：{metrics['total_runs']} 次配对运行满足预期；"
        f"V1 工具 {metrics['v1_tool_requests']}/{metrics['v1_tool_executions']}，"
        f"V2 工具 {metrics['v2_tool_requests']}/{metrics['v2_tool_executions']}"
    )
    print(
        f"恢复重复执行：V1={metrics['v1_replayed']}，"
        f"V2={metrics['v2_replayed']}"
    )
    print("口径：以上均为确定性离线 Mock，不代表真实模型质量或线上延迟。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
