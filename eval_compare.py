"""V1/V2 对比评测：python eval_compare.py --mode offline --repeats 3

--mode live 显式选择真实 API；仅运行可比较的三项业务场景。此脚本独占进程，
V1 的依赖替换在每次运行后恢复，不应作为并发 Web handler 使用。
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import statistics
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from eval_scenarios import CASES, ScenarioClient


@contextmanager
def evaluation_environment(mode):
    """离线模式在导入业务模块之前阻止 .env、socket 与云追踪。"""
    with ExitStack() as stack:
        if mode == "offline":
            stack.enter_context(patch("dotenv.load_dotenv"))
            stack.enter_context(patch.dict(os.environ, {
                "OPENAI_API_KEY": "offline-eval", "OPENAI_BASE_URL": "https://offline.invalid/v1",
                "LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "false",
            }))
            stack.enter_context(patch("socket.socket.connect", side_effect=AssertionError("offline: network forbidden")))
        yield


def protocol_errors(messages):
    """检查每个工具结果 ID 对应前一条 assistant 声明，模型请求不含悬空调用。"""
    pending = set()
    errors = []
    for message in messages:
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                errors.append("unmatched tool response")
            pending.discard(call_id)
        else:
            if pending:
                errors.append("missing tool response")
            calls = message.get("tool_calls", [])
            ids = [c["id"] for c in calls]
            if len(ids) != len(set(ids)):
                errors.append("duplicate tool_call_id in batch")
            pending = set(ids)
    if pending:
        errors.append("unresolved tool calls before model request")
    return errors


class Recorder:
    """并行安全的观测器：SDK 请求、usage 和真实 tool_runner 分开计数。"""
    def __init__(self, client, case, mode):
        self.client, self.case, self.mode = client, case, mode
        self.models, self.tools, self.protocol = [], [], []
        self.lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        errors = protocol_errors(kwargs["messages"])
        event = dict(start=time.perf_counter(), tool_requests=[], usage=None, error=None)
        with self.lock:
            self.protocol.extend(errors)
            self.models.append(event)
        try:
            response = self.client.chat.completions.create(**kwargs)
            event["tool_requests"] = [c.function.name for c in response.choices[0].message.tool_calls or []]
            usage = getattr(response, "usage", None)
            if usage is not None:
                event["usage"] = {key: getattr(usage, key, None) for key in (
                    "prompt_tokens", "completion_tokens", "total_tokens")}
            return response
        except Exception as exc:
            event["error"] = type(exc).__name__
            raise
        finally:
            event["end"] = time.perf_counter()

    def run_tool(self, name, arguments):
        from governance import ExecutionTimeout
        from tools import run_tool
        event = dict(name=name, arguments=dict(arguments), start=time.perf_counter(), ok=False)
        with self.lock:
            self.tools.append(event)
            ordinal = len(self.tools)
        try:
            # 评测数据只需要计算/时间；模型意外请求读文件也不会读取真实用户数据。
            if name not in {"calculator", "get_current_time"}:
                raise ValueError("tool not permitted in benchmark")
            if ordinal == 1 and self.case.get("fault"):
                if self.case["fault"] == "timeout":
                    raise ExecutionTimeout("injected timeout")
                raise RuntimeError("injected temporary failure")
            result = ("2026-01-01 00:00:00" if self.mode == "offline" and name == "get_current_time"
                      else run_tool(name, arguments))
            event["ok"] = True
            return result
        except Exception as exc:
            event["error"] = type(exc).__name__
            raise
        finally:
            event["end"] = time.perf_counter()

    def tokens(self):
        keys = ("prompt_tokens", "completion_tokens", "total_tokens")
        # 缺一轮 usage 就不能将部分总和冒充整轮成本；另报覆盖率。
        return {key: sum(e["usage"][key] for e in self.models)
                if self.models and all(e["usage"] and type(e["usage"].get(key)) is int for e in self.models)
                else None for key in keys}


def score(case, answer, stop_reason, recorder):
    reasons = []
    expected_stop = case.get("expected_stop", "completed")
    if stop_reason != expected_stop:
        reasons.append(f"stop: expected {expected_stop}, got {stop_reason}")
    if expected_stop != "error" and (not answer.strip() or case["answer"] not in answer):
        reasons.append("answer oracle failed")
    if expected_stop == "error" and not (
        len(recorder.models) == 1 and recorder.models[0].get("error") == "RuntimeError"
    ):
        reasons.append("expected injected model failure was not observed")
    expected = Counter(case["expressions"])
    actual = Counter(e["arguments"].get("expression") for e in recorder.tools)
    expected_names = Counter({"get_current_time" if case["id"] == "loop_limit" else "calculator": len(case["expressions"])})
    if not case["expressions"]:
        expected_names = Counter()
    trajectory_errors = list(recorder.protocol)
    if actual != expected or Counter(e["name"] for e in recorder.tools) != expected_names:
        trajectory_errors.append("actual tool calls/arguments differ from oracle")
    expected_outcomes = ([False] + [True] * (len(case["expressions"]) - 1)
                         if case.get("fault") else [True] * len(case["expressions"]))
    if [e["ok"] for e in recorder.tools] != expected_outcomes:
        trajectory_errors.append("tool success/failure sequence differs from oracle")
    if case["mode"] == "multi":
        predecessors = [e for e in recorder.tools if e["arguments"].get("expression") in {"12*5", "7*8"}]
        joins = [e for e in recorder.tools if e["arguments"].get("expression") == "60+56"]
        if len(predecessors) != 2 or len(joins) != 1 or any(
            p["end"] > joins[0]["start"] or not p["ok"] for p in predecessors
        ):
            trajectory_errors.append("dependency join ran before successful predecessors")
    expected_requests = case.get("requested", len(case["expressions"]))
    if sum(len(e["tool_requests"]) for e in recorder.models) != expected_requests:
        trajectory_errors.append("proposed tool call count differs from scenario")
    reasons.extend(trajectory_errors)
    return not reasons, not trajectory_errors, reasons


def run_case(case, engine, mode):
    import agent
    import multiagent
    from langgraph_agent import AgentDependencies, build_agent_graph, invoke_agent
    from langgraph_multiagent import MultiAgentDependencies, build_multiagent_graph, invoke_multiagent
    from tools import TOOLS, IDEMPOTENT_TOOLS
    base = ScenarioClient(case) if mode == "offline" else agent.client
    recorder = Recorder(base, case, mode)
    answer, stop_reason, error = "", "completed", None
    started = time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(agent, "log_event"))
        stack.enter_context(patch.object(multiagent, "log_event"))
        try:
            if engine == "v1":
                stack.enter_context(patch.object(agent, "client", recorder))
                stack.enter_context(patch.object(agent, "run_tool", recorder.run_tool))
                answer = (multiagent.run_multiagent(case["goal"], max_steps=3) if case["mode"] == "multi"
                          else agent.run_agent(case["goal"], session_id=None, max_iterations=3))
                if "已强制停止" in answer:
                    stop_reason = "max_iterations"
            elif case["mode"] == "single":
                deps = AgentDependencies(recorder, agent.MODEL, TOOLS, recorder.run_tool,
                                         agent.SYSTEM_PROMPT, IDEMPOTENT_TOOLS,
                                         request_timeout_seconds=agent.MODEL_TIMEOUT_SECONDS)
                state = invoke_agent(case["goal"], max_iterations=3, graph=build_agent_graph(deps))
                answer, stop_reason = state["answer"], state["stop_reason"]
            else:
                deps = MultiAgentDependencies(recorder, agent.MODEL, recorder.run_tool,
                                              multiagent.ROLE_REGISTRY, IDEMPOTENT_TOOLS,
                                              executor_max_iterations=3,
                                              request_timeout_seconds=agent.MODEL_TIMEOUT_SECONDS)
                state = invoke_multiagent(case["goal"], max_tasks=3, graph=build_multiagent_graph(deps))
                answer, stop_reason = state["answer"], state["stop_reason"]
        except Exception as exc:
            stop_reason, error = "error", type(exc).__name__
    latency_ms = (time.perf_counter() - started) * 1000
    passed, trajectory_ok, reasons = score(case, answer, stop_reason, recorder)
    return dict(case_id=case["id"], engine=engine, mode=mode, kind=case["mode"],
                category="fault" if case.get("fault") or case.get("expected_stop") else "business",
                passed=passed, trajectory_ok=trajectory_ok, reasons=reasons,
                answer=answer, stop_reason=stop_reason, error=error, latency_ms=latency_ms,
                model_calls=len(recorder.models),
                tool_requests=sum(len(e["tool_requests"]) for e in recorder.models),
                tool_executions=len(recorder.tools), tokens=recorder.tokens(),
                usage_coverage=sum(bool(e["usage"]) for e in recorder.models) / len(recorder.models)
                if recorder.models else None, model_trace=recorder.models, tool_trace=recorder.tools)


def percentile95(values):
    return sorted(values)[max(0, math.ceil(len(values) * 0.95) - 1)]


def summarize(rows):
    summary = {}
    for engine in ("v1", "v2"):
        selected = [r for r in rows if r["engine"] == engine]
        business = [r for r in selected if r["category"] == "business"]
        times = [r["latency_ms"] for r in selected]
        total_tokens = [r["tokens"]["total_tokens"] for r in selected]
        summary[engine] = dict(
            runs=len(selected), expectation_pass_rate=sum(r["passed"] for r in selected) / len(selected),
            task_success_rate=sum(r["passed"] for r in business) / len(business) if business else None,
            trajectory_pass_rate=sum(r["trajectory_ok"] for r in selected) / len(selected),
            latency_median_ms=statistics.median(times), latency_p95_ms=percentile95(times),
            model_calls=sum(r["model_calls"] for r in selected),
            tool_requests=sum(r["tool_requests"] for r in selected),
            tool_executions=sum(r["tool_executions"] for r in selected),
            total_tokens=sum(total_tokens) if all(v is not None for v in total_tokens) else None,
        )
    return summary


def markdown_report(report):
    offline = report["mode"] == "offline"
    lines = ["# V1 / V2 对比评测", "",
             "**离线确定性 Mock 报告；不能代表真实模型质量、Token 成本或 API 延迟。**" if offline else
             "真实 API 小样本报告；评估固定用例，不代表通用能力或生产 SLA。",
             "", f"运行时间：{report['created_at']}；重复次数：{report['repeats']}。", "",
             "| 版本 | 业务通过率 | 预期行为通过率（含故障） | 轨迹通过率 | 中位/P95 ms | 模型调用 | 工具请求/执行 | Token |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for engine, s in report["summary"].items():
        lines.append(f"| {engine} | {s['task_success_rate']:.1%} | {s['expectation_pass_rate']:.1%} | "
                     f"{s['trajectory_pass_rate']:.1%} | {s['latency_median_ms']:.2f}/{s['latency_p95_ms']:.2f} | "
                     f"{s['model_calls']} | {s['tool_requests']}/{s['tool_executions']} | {s['total_tokens'] if s['total_tokens'] is not None else 'N/A'} |")
    lines += ["", "| 用例 | 版本 | 通过次数/样本数 | 平均实际工具执行 |", "|---|---|---:|---:|"]
    for case_id in dict.fromkeys(r["case_id"] for r in report["runs"]):
        for engine in ("v1", "v2"):
            rs = [r for r in report["runs"] if r["case_id"] == case_id and r["engine"] == engine]
            lines.append(f"| {case_id} | {engine} | {sum(r['passed'] for r in rs)}/{len(rs)} | {statistics.mean(r['tool_executions'] for r in rs):.1f} |")
    recovery = report.get("recovery")
    if recovery:
        lines += ["", "## 进程恢复实验（独立离线实验）", "",
                  "先执行同一计划的 task 1/2 并退出，再启动新 Python 进程。V1 从头重新运行，V2 从 SQLite 继续。",
                  "每个 task 的工具参数不同，台账按参数判重；只验证提交后的波次，不模拟写操作提交窗口中的断电。", "",
                  "| 版本 | 最终完成 | 恢复成功/实验次数 | 重复工具执行 |", "|---|---|---:|---:|"]
        for engine, r in recovery.items():
            lines.append(f"| {engine} | {r['completed']} | {int(r['recovered_without_replay'])}/1 | {r['duplicate_executions']} |")
    lines += ["", "## 统计口径", "",
              "- 业务通过率包含答案与轨迹约束；关键词检查不是语义裁判。模型异常/循环上限/注入故障单列预期行为，不能算成业务成功。",
              "- 轨迹检查工具名、参数、次数、tool_call_id 回填配对；多 Agent 合流检查前置工具完成早于下游工具开始，允许同层乱序。",
              "- 实际执行在 tool_runner 边界计数（含失败尝试）；模型提出但命中缓存的调用只计请求次数。",
              "- 所有 Planner/Executor/Summarizer 请求通过同一代理计数。usage 任一轮缺失，整次 Token 为 null/N/A，禁止当成 0。",
              "- 延迟包含每次建图与执行，排除模块导入、恢复子进程启动和报告写入；P95 用 nearest-rank，小样本仅用于本机参考。",
              "- 两版交替先后运行；共享用例、模型、工具和参考答案，多 Agent 提示与编排方式保留原实现差异。",
              "- eval 使用进程级依赖替换，禁止与业务流量在同一进程中并行；原始 JSON 包含完整逐次指标和执行轨迹。", ""]
    failures = [r for r in report["runs"] if not r["passed"]]
    if failures:
        lines += ["## 失败项", ""] + [f"- {r['case_id']}/{r['engine']}: {r['reasons']}" for r in failures]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, help="报告前缀（生成 .json 和 .md）")
    parser.add_argument("--skip-recovery", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output = args.output or Path(__file__).parent / f"reports/{args.mode}-eval"
    with evaluation_environment(args.mode):
        from governance import atomic_write_json, atomic_write_text
        cases = [c for c in CASES if args.mode == "offline" or c.get("live")]
        rows = []
        for repeat in range(args.repeats):
            for case in cases:
                for engine in (("v1", "v2") if repeat % 2 == 0 else ("v2", "v1")):
                    row = run_case(case, engine, args.mode)
                    row["repeat"] = repeat + 1
                    rows.append(row)
                    print(f"{'PASS' if row['passed'] else 'FAIL'} {engine} {case['id']} ({repeat + 1})", flush=True)
        recovery = None
        if not args.skip_recovery:
            from eval_recovery import compare_recovery
            print("RUN offline process-restart experiment", flush=True)
            recovery = compare_recovery()
        report = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                      mode=args.mode, repeats=args.repeats, runs=rows, summary=summarize(rows),
                      recovery=recovery, versions={p: importlib.metadata.version(p) for p in
                          ("langgraph", "langgraph-checkpoint-sqlite", "openai")})
        import agent
        import multiagent
        root = Path(__file__).parent
        report["environment"] = dict(python=platform.python_version(), platform=platform.platform(),
            model="ScenarioClient (no API)" if args.mode == "offline" else agent.MODEL,
            max_iterations=3, max_tasks=3, v1_max_parallel=multiagent.MAX_PARALLEL, v2_max_concurrency=4,
            model_request_timeout=agent.MODEL_TIMEOUT_SECONDS)
        report["cases"] = cases
        report["source_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
            "agent.py", "multiagent.py", "langgraph_agent.py", "langgraph_multiagent.py",
            "checkpointing.py", "tools.py", "governance.py", "eval_compare.py", "eval_scenarios.py", "eval_recovery.py")}
        atomic_write_json(args.output.with_suffix(".json"), report, indent=2)
        atomic_write_text(args.output.with_suffix(".md"), markdown_report(report))
        print(f"Report: {args.output.with_suffix('.md').resolve()}")
        return 0 if all(r["passed"] for r in rows) and (not recovery or recovery["v2"]["recovered_without_replay"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
