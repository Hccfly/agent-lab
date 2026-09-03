"""eval 评测:真实调用模型,给 agent 一组 (问题, 期望关键词/期望工具) 用例自动打分。

这是与 tests/ 单元测试互补的质量层:
- tests/*.py      不调 API,monkeypatch 模拟模型,验证"逻辑正确性";
- evaluate.py     真调 API,验证"端到端行为"——会不会用对工具、答案对不对。

用法(需 .env 配好 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL):
    python evaluate.py            # 跑全部用例
    python evaluate.py 计算        # 按名字或模式(single/multi)过滤

返回码:全部通过 0,否则 1(便于 CI 接入)。
"""
import logging
import sys
import time

import agent
from agent import run_agent
from multiagent import run_multiagent
from log import setup_logging

# 用例结构:
#   name      展示名
#   mode      single(单 agent)/ multi(多 agent)
#   goal      真实发给模型的问题
#   tool      期望被调用的工具名;None = 期望本轮不调用任何工具
#   keywords  期望最终答案包含的关键词(大小写不敏感),可空
#
# 设计:故意放"agent 该用工具却可能心算/乱猜"的用例——不用工具就 FAIL,
# 才是 eval 该抓的质量问题,而不是只测"能跑通"。
CASES = [
    {
        "name": "单agent·精确计算 127*53",
        "mode": "single",
        "goal": "请调用计算工具算出 127 乘以 53,只告诉我结果数字。",
        "tool": "calculator",
        "keywords": ["6731"],
    },
    {
        "name": "单agent·实时时间(应调时间工具)",
        "mode": "single",
        "goal": "现在几点?请调用时间工具查询实时日期时间,再回答我。",
        "tool": "get_current_time",
        "keywords": [],
    },
    {
        "name": "单agent·纯对话(不应调用任何工具)",
        "mode": "single",
        "goal": "这是一个普通对话,不需要调用任何工具。请用一句话介绍你自己。",
        "tool": None,
        "keywords": [],
    },
    {
        "name": "多agent·拆解计算 9*9",
        "mode": "multi",
        "goal": "请算一下 9 乘以 9,告诉我结果。",
        "tool": "calculator",
        "keywords": ["81"],
    },
    {
        "name": "多agent·文件角色读 tools.py",
        "mode": "multi",
        "goal": "读文件 D:/实习/agent-lab/tools.py,告诉我里面用 _tool 注册了哪几个工具函数。",
        "tool": "read_file",
        "keywords": ["工具"],
    },
]


def check_case(case: dict, answer: str, calls: list[str]) -> tuple[bool, list[str]]:
    """判一次执行是否达标。answer=最终答案;calls=执行期间实际调用的工具名。

    返回 (是否通过, 失败原因列表)。纯函数,不调 API——单测直接覆盖。
    """
    reasons: list[str] = []
    ans = (answer or "").strip()
    if not ans:
        reasons.append("回答为空")
    expected = case.get("tool")
    if expected is None:
        if calls:
            reasons.append(f"预期不调用工具,实际调用了 {sorted(set(calls))}")
    elif expected not in calls:
        reasons.append(f"预期调用工具 {expected!r},实际 {sorted(set(calls)) or '(无调用)'}")
    low = ans.lower()
    for kw in case.get("keywords", []):
        if kw.lower() not in low:
            reasons.append(f"答案缺少关键内容 {kw!r}")
    return (not reasons), reasons


def run_case(case: dict) -> dict:
    """真实执行一次(打 API),返回结构化结果。

    执行期间用一个 spy 包住 agent.log_event,收集本用例实际调用的工具名
    (工具调用全在 agent.py 作用域内,经 log_event("tool_call", ..., tool=...) 落日志)。
    """
    calls: list[str] = []
    real = agent.log_event

    def spy(event: str, msg: str = "", level: int = logging.INFO, **fields) -> None:
        if event == "tool_call":
            calls.append(fields.get("tool"))
        return real(event, msg, level=level, **fields)

    agent.log_event = spy
    error = None
    t0 = time.monotonic()
    try:
        if case.get("mode", "single") == "multi":
            answer = run_multiagent(case["goal"], stream=False)
        else:
            answer = run_agent(case["goal"], session_id=None, stream=False)
    except Exception as exc:  # 单用例失败不应中断整轮 eval
        answer, error = "", f"{type(exc).__name__}: {exc}"
    finally:
        agent.log_event = real
    latency = time.monotonic() - t0

    ok, reasons = check_case(case, answer, calls)
    return {
        "name": case["name"],
        "mode": case.get("mode", "single"),
        "goal": case["goal"],
        "tool": case.get("tool"),
        "keywords": case.get("keywords", []),
        "ok": ok,
        "answer": answer,
        "calls": calls,
        "latency": latency,
        "reasons": reasons,
        "error": error,
    }


def main() -> int:
    # eval 的模型交互日志单独落文件,不刷终端(打分报告本身就是终端输出)。
    setup_logging("logs/eval.jsonl", console=False)

    only = sys.argv[1] if len(sys.argv) > 1 else None
    cases = CASES
    if only:
        cases = [c for c in CASES if only in c["name"] or only == c.get("mode")]
    if not cases:
        print(f"没有匹配 {only!r} 的用例")
        return 1

    print(f"\n=== Agent eval(真实 API,共 {len(cases)} 个用例)===")
    passed = 0
    for c in cases:
        r = run_case(c)
        tag = "PASS" if r["ok"] else "FAIL"
        passed += int(bool(r["ok"]))
        print(f"\n[{tag}] {r['name']}  calls={r['calls']}  耗时 {r['latency']:.1f}s")
        if r["error"]:
            print(f"      - 异常: {r['error']}")
        for why in r["reasons"]:
            print(f"      - {why}")
        if not r["ok"]:
            print(f"      答案: {r['answer']!r}"[:160])

    total = len(cases)
    print(f"\n通过 {passed}/{total}({passed / total * 100:.0f}%)")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
