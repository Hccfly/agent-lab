"""eval 层冒烟测试:不调真实 API,monkeypatch 模拟模型。
覆盖:check_case 打分纯函数 + run_case 的工具调用收集(spy)+ 单/多 agent 链路。
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import multiagent
from evaluate import check_case, run_case


class _FakeTool:
    def __init__(self, name, arguments):
        self.id = "call_1"
        self.type = "function"
        self.function = SimpleNamespace(name=name, arguments=arguments)


class _FakeMsg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, mode="json", exclude_none=True):
        d = {"role": "assistant"}
        if self.content:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": t.id,
                    "type": "function",
                    "function": {"name": t.function.name, "arguments": t.function.arguments},
                }
                for t in self.tool_calls
            ]
        return d


def _resp(msg):
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


# --- 测试 1:check_case 打分纯函数 ---
# 1a. 全过:关键词命中 + 期望工具被调
case_ok = {"tool": "calculator", "keywords": ["6731"]}
ok, reasons = check_case(case_ok, "127*53 = 6731", ["calculator"])
assert ok and not reasons, (ok, reasons)
print("OK  check_case:关键词 + 期望工具都命中 -> 通过")

# 1b. 该用工具却没用(心算) -> 失败
ok, reasons = check_case(case_ok, "127*53 = 6731", [])
assert not ok and any("预期调用工具 'calculator'" in r for r in reasons), reasons
print("OK  check_case:该调用工具却心算 -> 失败")

# 1c. 答案缺关键词 -> 失败
ok, reasons = check_case(case_ok, "我算好了", ["calculator"])
assert not ok and any("缺少关键内容" in r for r in reasons), reasons
print("OK  check_case:答案缺关键词 -> 失败")

# 1d. tool=None(期望不调工具)但实际调了 -> 失败
ok, reasons = check_case({"tool": None}, "你好", ["get_current_time"])
assert not ok and any("预期不调用工具" in r for r in reasons), reasons
print("OK  check_case:纯对话却调了工具 -> 失败")

# 1e. 空答案 -> 失败
ok, reasons = check_case(case_ok, "   ", ["calculator"])
assert not ok and any("回答为空" in r for r in reasons), reasons
print("OK  check_case:空回答 -> 失败")


# --- 测试 2:run_case 单 agent,spy 收集工具调用并打分 ---
def fake_create_single(**kwargs):
    msgs = kwargs.get("messages", [])
    if any(m.get("role") == "tool" for m in msgs):
        return _resp(_FakeMsg(content="结果是 6731"))
    return _resp(_FakeMsg(tool_calls=[_FakeTool("calculator", '{"expression": "127*53"}')]))


with patch("agent.client.chat.completions.create", fake_create_single):
    r = run_case(
        {"name": "t", "mode": "single", "goal": "算 127*53",
         "tool": "calculator", "keywords": ["6731"]}
    )

assert r["ok"], r
assert r["calls"] == ["calculator"], f"应收集到实际工具调用: {r['calls']}"
assert "6731" in r["answer"] and r["error"] is None, r
print("OK  run_case(单agent):spy 收集到 calculator,打分通过")
print("    calls =", r["calls"], "| answer =", r["answer"])


# --- 测试 3:run_case 单 agent,纯对话(不该调工具)链路 ---
def fake_create_nofunc(**kwargs):
    return _resp(_FakeMsg(content="我是助手,不需要工具。"))


with patch("agent.client.chat.completions.create", fake_create_nofunc):
    r = run_case(
        {"name": "t", "mode": "single", "goal": "介绍你自己",
         "tool": None, "keywords": []}
    )

assert r["ok"], r
assert r["calls"] == [], r
print("OK  run_case(单agent):纯对话不调工具 -> 通过")


# --- 测试 4:run_case 多 agent 链路(Planner 拆解 + Executor 被 mock) ---
def fake_executor(user_input, **kwargs):
    return "81"


def fake_create_multi(**kwargs):
    msgs = kwargs.get("messages", [])
    if msgs[0]["content"] == multiagent.PLANNER_SYSTEM:
        has_results = any("已完成子任务的结果" in m.get("content", "") for m in msgs[1:])
        if has_results:
            return _resp(_FakeMsg(content="最终答案: 81"))
        return _resp(
            _FakeMsg(content='{"tasks": [{"task_id": 1, "agent": "calculator", '
                             '"description": "算 9*9", "reason": "r"}]}')
        )
    return _resp(_FakeMsg(content=""))


with patch("agent.client.chat.completions.create", fake_create_multi), \
     patch("multiagent.agent.run_agent", side_effect=fake_executor):
    r = run_case(
        {"name": "t", "mode": "multi", "goal": "算 9*9",
         "tool": None, "keywords": ["81"]}
    )

assert r["ok"], r
assert "81" in r["answer"] and r["error"] is None, r
print("OK  run_case(多agent):Planner/Executor 链路跑通,最终答案含 81")


# --- 测试 5:run_case 内部异常被捕获,不中断 ---
def fake_create_crash(**kwargs):
    raise RuntimeError("api down")


with patch("agent.client.chat.completions.create", fake_create_crash):
    r = run_case(
        {"name": "t", "mode": "single", "goal": "hi", "tool": None, "keywords": []}
    )

assert not r["ok"], r
assert r["error"] and "api down" in r["error"], r
assert r["calls"] == [], r
print("OK  run_case:模型调用抛异常 -> 记录 error,不中断整轮")


print("\n全部通过")
