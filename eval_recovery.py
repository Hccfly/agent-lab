"""子进程恢复实验；只使用 Mock 模型和本地计算器。"""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

from eval_compare import evaluation_environment
from eval_scenarios import CASES, ScenarioClient


class SimulatedStop(BaseException):
    """已完成前两项后退出当前运行，不属于业务工具异常。"""


def worker(engine, phase, directory):
    with evaluation_environment("offline"):
        import agent
        import multiagent
        from checkpointing import PersistentMultiAgent
        from governance import atomic_write_json, locked_path
        from langgraph_multiagent import MultiAgentDependencies
        from tools import IDEMPOTENT_TOOLS, run_tool
        case = next(c for c in CASES if c["id"] == "multi_diamond")
        client = ScenarioClient(case)
        ledger_path = directory / "ledger.json"

        def recorded_tool(name, arguments):
            result = run_tool(name, arguments)
            with locked_path(ledger_path):
                ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else []
                ledger.append(dict(expression=arguments["expression"], pid=os.getpid(), phase=phase))
                atomic_write_json(ledger_path, ledger)
            return result

        completed = False
        with patch.object(agent, "log_event"), patch.object(multiagent, "log_event"):
            if engine == "v1":
                original = multiagent._execute_one

                def execute(task, *args, **kwargs):
                    if phase == "start" and task["task_id"] == 3:
                        raise SimulatedStop()
                    return original(task, *args, **kwargs)

                with patch.object(agent, "client", client), patch.object(agent, "run_tool", recorded_tool), patch.object(
                    multiagent, "_execute_one", execute
                ):
                    try:
                        answer = multiagent.run_multiagent(case["goal"], max_steps=3)
                        completed = "116" in answer
                    except SimulatedStop:
                        pass
            else:
                deps = MultiAgentDependencies(client, "offline", recorded_tool,
                                              multiagent.ROLE_REGISTRY, IDEMPOTENT_TOOLS)
                with PersistentMultiAgent(directory / "checkpoint.sqlite3", deps,
                                          interrupt_after=["executor"] if phase == "start" else None) as runner:
                    run = runner.start(case["goal"], "restart-eval", max_tasks=3) if phase == "start" else runner.resume("restart-eval")
                    completed = run.completed and run.state["stop_reason"] == "completed"
        print(json.dumps(dict(pid=os.getpid(), completed=completed)))


def compare_recovery():
    results = {}
    script = str(Path(__file__).resolve())
    for engine in ("v1", "v2"):
        with tempfile.TemporaryDirectory(prefix="agent-eval-recovery-") as temp:
            directory = Path(temp)
            phases = []
            for phase in ("start", "resume"):
                child = subprocess.run([sys.executable, "-X", "utf8", script, engine, phase, temp],
                                       cwd=Path(__file__).parent, capture_output=True, text=True,
                                       encoding="utf-8", timeout=60, check=True)
                phases.append(json.loads(child.stdout))
                if phase == "start":
                    initial = json.loads((directory / "ledger.json").read_text(encoding="utf-8"))
            ledger = json.loads((directory / "ledger.json").read_text(encoding="utf-8"))
            counts = Counter(item["expression"] for item in ledger)
            initial_ok = Counter(x["expression"] for x in initial) == Counter(["12*5", "7*8"])
            duplicates = sum(max(0, count - 1) for count in counts.values())
            results[engine] = dict(completed=phases[-1]["completed"],
                                   recovered_without_replay=initial_ok and phases[-1]["completed"] and
                                       counts == Counter(["12*5", "7*8", "60+56"]),
                                   initial_boundary_verified=initial_ok,
                                   duplicate_executions=duplicates, phases=phases, tool_ledger=ledger)
    return results


if __name__ == "__main__":
    worker(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
