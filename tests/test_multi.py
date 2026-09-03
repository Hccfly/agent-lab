"""多 agent 层冒烟测试:不调真实 API,monkeypatch 模拟模型。"""
import sys
import time
from pathlib import Path

# 让脚本能从项目根目录导入模块(multiagent 等)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import patch

import multiagent
from multiagent import run_multiagent


class FakeMsg:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None

    def model_dump(self, mode="python", exclude_none=True):
        return {"role": "assistant", "content": self.content}


class FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": FakeMsg(content)})()]


# --- 测试 1:Planner 拆解 -> Executor 执行 -> 汇总最终答案 ---
# 模拟对话:第一轮 Planner 输出拆解 JSON,第二轮输出最终答案
calls = []


def fake_create(**kwargs):
    msgs = kwargs.get("messages", [])
    # 判断是 Planner 调用还是 Executor 调用:Planner 的 system 是 PLANNER_SYSTEM
    is_planner = msgs[0]["content"] == multiagent.PLANNER_SYSTEM
    if is_planner:
        # 第一轮:拆解;之后:有结果了就输出最终答案
        has_results = any("已完成子任务的结果" in m.get("content", "") for m in msgs[1:])
        if has_results:
            return FakeResp("完成。现在 2026-09-01,明天是 9 月 2 日。")
        return FakeResp('{"tasks": [{"task_id": 1, "description": "算 3+4", "reason": "简单"}]}')
    else:
        # Executor:返回子任务结果
        return FakeResp("子任务结果: 7")


with patch.object(multiagent.agent.client.chat.completions, "create", fake_create), \
     patch("multiagent.agent.run_agent", return_value="子任务结果: 7"):
    out = run_multiagent("明天几号", max_steps=5)

assert "9 月 2 日" in out, f"多 agent 未收敛到最终答案: {out!r}"
print("OK  多 agent:拆解 -> 执行 -> 汇总最终答案")
print("    输出:", out)


# --- 测试 2:拆解失败时兜底为直接执行 ---
def fake_create_fail(**kwargs):
    msgs = kwargs.get("messages", [])
    if msgs[0]["content"] == multiagent.PLANNER_SYSTEM:
        return FakeResp("我不会拆解,直接说:结果是 42")  # 非 JSON
    return FakeResp("")


with patch.object(multiagent.agent.client.chat.completions, "create", fake_create_fail), \
     patch("multiagent.agent.run_agent", return_value="兜底执行结果") as m:
    out = run_multiagent("随便一个任务", max_steps=3)
assert "直接说:结果是 42" in out, f"非 JSON 应直接作为最终答案: {out!r}"
print("OK  最终答案(非 JSON)直接返回")


# --- 测试 3:JSON 提取稳健性(容忍代码块包裹) ---
s = '```json\n{"tasks": [{"task_id": 1, "description": "a", "reason": "b"}]}\n```'
parsed = multiagent._extract_json(s)
assert parsed["tasks"][0]["description"] == "a"
print("OK  _extract_json 容忍 ```json 包裹")


# --- 测试 8:Planner 用 ```json 代码块包裹拆解 -> 应解析执行,不当最终答案返回 ---
fenced_calls = []


def fake_executor_fenced(user_input, **kwargs):
    fenced_calls.append(user_input)
    return "文件已读取,3 个依赖"


def fake_create_fenced(**kwargs):
    msgs = kwargs.get("messages", [])
    if msgs[0]["content"] == multiagent.PLANNER_SYSTEM:
        has_results = any("已完成子任务的结果" in m.get("content", "") for m in msgs[1:])
        if has_results:
            return FakeResp("最终答案:共 3 个依赖")
        # 模型无视"不要 markdown",用代码块包 JSON 拆解
        return FakeResp('```json\n{"tasks": [{"task_id": 1, "agent": "file", '
                        '"description": "读取文件", "reason": "fenced"}]}\n```')
    return FakeResp("")


with patch.object(multiagent.agent.client.chat.completions, "create", fake_create_fenced), \
     patch("multiagent.agent.run_agent", side_effect=fake_executor_fenced):
    out = run_multiagent("读文件", max_steps=4)

assert len(fenced_calls) == 1, "代码块包裹的拆解应被执行,而不是被当最终答案返回"
assert "3 个依赖" in out
print("OK  代码块包裹的拆解被解析执行,不再原样返回原始 JSON")


# --- 测试 4:任务带 agent 归属 -> 路由到专用 Executor(工具子集 + 角色提示) ---
seen = {}


def fake_executor(user_input, **kwargs):
    seen["tools"] = kwargs.get("tools")
    seen["system"] = kwargs.get("system_prompt")
    return "计算完成: 7"


def fake_create_roles(**kwargs):
    msgs = kwargs.get("messages", [])
    if msgs[0]["content"] == multiagent.PLANNER_SYSTEM:
        has_results = any("已完成子任务的结果" in m.get("content", "") for m in msgs[1:])
        if has_results:
            return FakeResp("结果正确:7")
        return FakeResp(
            '{"tasks": [{"task_id": 1, "agent": "calculator", '
            '"description": "算 3+4", "reason": "纯数值计算"}]}'
        )
    return FakeResp("")


with patch.object(multiagent.agent.client.chat.completions, "create", fake_create_roles), \
     patch("multiagent.agent.run_agent", side_effect=fake_executor):
    out = run_multiagent("算 3+4", max_steps=4)

calc_tools = multiagent.ROLE_REGISTRY["calculator"]["tools"]
assert seen["tools"] == calc_tools, f"calculator 角色应拿到计算工具子集: {seen['tools']}"
assert len(seen["tools"]) == 1 and seen["tools"][0]["function"]["name"] == "calculator"
assert "计算专员" in seen["system"], "calculator 角色提示应声明身份"
assert "7" in out
print("OK  路由:任务标 agent=calculator -> 计算专员执行(仅 calculator 工具)")


# --- 测试 5:缺省 agent 字段 / 未知角色 -> 兜底 general(全量工具) ---
seen2 = {}


def fake_executor2(user_input, **kwargs):
    seen2.setdefault("roles", []).append(kwargs.get("system_prompt"))
    seen2.setdefault("tools_len", []).append(len(kwargs.get("tools")))
    return "done"


def fake_create_mixed(**kwargs):
    msgs = kwargs.get("messages", [])
    if msgs[0]["content"] == multiagent.PLANNER_SYSTEM:
        has_results = any("已完成子任务的结果" in m.get("content", "") for m in msgs[1:])
        if has_results:
            return FakeResp("最终: 完成")
        # 两个任务:一个没标 agent,一个标了不存在的角色
        return FakeResp(
            '{"tasks": [{"task_id": 1, "description": "无归属任务"}, '
            '{"task_id": 2, "agent": "hacker", "description": "未知角色任务"}]}'
        )
    return FakeResp("")


with patch.object(multiagent.agent.client.chat.completions, "create", fake_create_mixed), \
     patch("multiagent.agent.run_agent", side_effect=fake_executor2):
    run_multiagent("测试", max_steps=4)

assert seen2["tools_len"] == [len(multiagent.ROLE_REGISTRY["general"]["tools"])] * 2, seen2
assert "通用专员" in seen2["roles"][0] and "通用专员" in seen2["roles"][1]
print("OK  兜底:缺省 agent / 未知角色都路由到 general(全量工具)")


# --- 测试 6:有依赖的子任务——后置任务会收到前置任务的真实结果 ---
inject_prompts = []


def fake_executor_inject(user_input, **kwargs):
    inject_prompts.append(user_input)
    return "7" if len(inject_prompts) == 1 else "8"


def fake_create_dep(**kwargs):
    msgs = kwargs.get("messages", [])
    if msgs[0]["content"] == multiagent.PLANNER_SYSTEM:
        has_results = any("已完成子任务的结果" in m.get("content", "") for m in msgs[1:])
        if has_results:
            return FakeResp("最终答案: 8")
        # 任务 2 用 depends_on 显式声明依赖任务 1:Orchestrator 把任务 2 排到任务 1
        # 完成之后的波,并在它的上下文里附上任务 1 的真实结果
        return FakeResp(
            '{"tasks": [{"task_id": 1, "agent": "calculator", "description": "算 3+4", '
            '"reason": "数值计算"}, {"task_id": 2, "agent": "general", '
            '"description": "基于上一步的结果再加 1", "reason": "依赖任务1", '
            '"depends_on": [1]}]}'
        )
    return FakeResp("")


with patch.object(multiagent.agent.client.chat.completions, "create", fake_create_dep), \
     patch("multiagent.agent.run_agent", side_effect=fake_executor_inject):
    out = run_multiagent("算依赖", max_steps=4)

# 任务 1 没有前置,任务 2 分派时应附上任务 1 的"7"
assert "任务 1" not in inject_prompts[0], "第一个任务不应有前置结果"
assert "7" in inject_prompts[1], f"任务 2 应收到任务 1 的真实结果(7): {inject_prompts[1]!r}"
# 注入措辞应强制:内容已提供,不要自己读取/索要路径
assert "不要尝试自己读取文件" in inject_prompts[1], "注入措辞应禁止 Executor 重新读取"
assert "直接基于这些内容完成" in inject_prompts[1]
assert "8" in out
print("OK  依赖任务:后置子任务收到前置真实结果,且措辞禁止重新读取/索要路径")


# --- 测试 7:有依赖任务分批——Planner 提示不一次拆完依赖 ---
dep_hint = multiagent.PLANNER_SYSTEM
assert "依赖分批" in dep_hint, "Planner 提示应含依赖分批规则"
assert "不要把这种有依赖的任务放在同一批" in dep_hint
print("OK  Planner 提示包含依赖分批规则")

# --- 测试 9:Executor 返回失败信号 -> Orchestrator 自动用 general 重派一次并收敛 ---
retry_log = []
GENERAL_TOOLS = multiagent.ROLE_REGISTRY["general"]["tools"]
GENERAL_LEN = len(GENERAL_TOOLS)


def fake_executor_retry(user_input, **kwargs):
    tools = kwargs.get("tools")
    # 第一次(专用角色,工具少)总是失败;重派(general 全量)才成功
    if len(tools) < GENERAL_LEN:
        retry_log.append(("fail", user_input, tools))
        return "我无法完成这个子任务:需要读取文件内容,但读取文件超出我的能力。"
    retry_log.append(("success", user_input, tools))
    return "文件内容: 一共 3 行。"


def fake_create_retry(**kwargs):
    msgs = kwargs.get("messages", [])
    if msgs[0]["content"] == multiagent.PLANNER_SYSTEM:
        has_results = any("已完成子任务的结果" in m.get("content", "") for m in msgs[1:])
        if has_results:
            return FakeResp("最终答案: 文件一共 3 行。")
        return FakeResp(
            '{"tasks": [{"task_id": 1, "agent": "calculator", '
            '"description": "读取文件内容", "reason": "测试"}]}'
        )
    return FakeResp("")


with patch.object(multiagent.agent.client.chat.completions, "create", fake_create_retry), \
     patch("multiagent.agent.run_agent", side_effect=fake_executor_retry):
    out = run_multiagent("读文件", max_steps=4)

assert len(retry_log) == 2, f"失败后应自动重派一次,共 2 次执行: {retry_log}"
assert retry_log[0][0] == "fail" and retry_log[1][0] == "success", retry_log
# 重派用 general 全量工具,且 prompt 带回失败原因 + 强措辞
assert len(retry_log[1][2]) == GENERAL_LEN, "重派应使用 general 全量工具"
assert "无法完成" in retry_log[1][1], "重派 prompt 应包含上一次的失败原因"
assert "重新完成它" in retry_log[1][1], "重派 prompt 应强令重新完成"
assert "read_file" in retry_log[1][1]
assert "3 行" in out
print("OK  失败自愈:Executor 失败 -> 自动用 general 重派 -> 收敛")
print("    重派 prompt 含失败原因与 read_file 提示:", "无法完成" in retry_log[1][1])


# --- 测试 10:重派后仍失败 -> 只重试一次(不无限循环),失败结果回填 Planner ---
retry_count = []


def fake_executor_always_fail(user_input, **kwargs):
    retry_count.append(kwargs.get("tools"))
    return "我无法完成,因为缺少前置任务的结果。"


def fake_create_still_fail(**kwargs):
    msgs = kwargs.get("messages", [])
    if msgs[0]["content"] == multiagent.PLANNER_SYSTEM:
        has_results = any("已完成子任务的结果" in m.get("content", "") for m in msgs[1:])
        if has_results:
            return FakeResp("最终答案: 子任务失败,已记录原因。")
        return FakeResp(
            '{"tasks": [{"task_id": 1, "agent": "general", '
            '"description": "分析数据", "reason": "测试"}]}'
        )
    return FakeResp("")


with patch.object(multiagent.agent.client.chat.completions, "create", fake_create_still_fail), \
     patch("multiagent.agent.run_agent", side_effect=fake_executor_always_fail):
    out = run_multiagent("测试", max_steps=4)

assert len(retry_count) == 2, f"重试后仍失败应只重派一次(不无限循环),共 2 次: {len(retry_count)}"
assert len(retry_count[1]) == GENERAL_LEN, "第二次重派仍走 general 全量工具"
assert "任务失败" in out
print("OK  失败自愈:重派仍失败 -> 只重试一次,失败原因回填 Planner")


# --- 测试 11:_topological_waves 拓扑分层 + 并行窗口重叠 + depends_on 注入 ---
# 11a. 纯函数:分层正确(独立同波、依赖后移)
w = multiagent._topological_waves(
    [{"task_id": 1, "depends_on": []}, {"task_id": 2}, {"task_id": 3, "depends_on": [1, 2]}],
    set(),
)
assert [sorted(t["task_id"] for t in wv) for wv in w] == [[1, 2], [3]], w
w2 = multiagent._topological_waves(
    [{"task_id": 1}, {"task_id": 2, "depends_on": [1]}], set([5])
)
# 依赖 1 未完成 -> 1 先进波,2 后进;外部已完成 5 不阻塞任何人
assert [t["task_id"] for t in w2[0]] == [1] and [t["task_id"] for t in w2[1]] == [2], w2
print("OK  _topological_waves:独立任务同波可并行,depends_on 任务后移")

# 11b. 集成:两个独立任务(波1)真实并行(time 窗重叠),依赖任务(波2)注入前置结果
par_events = []  # (task, phase, t)


def fake_executor_par(user_input, **kwargs):
    t = time.time()
    if "1+1" in user_input:
        par_events.append(("t1", "start", t)); time.sleep(0.08)
        par_events.append(("t1", "end", time.time())); return "2"
    if "2+2" in user_input:
        par_events.append(("t2", "start", t)); time.sleep(0.08)
        par_events.append(("t2", "end", time.time())); return "4"
    # 任务 3(depends_on [1,2]):应等任务 1、2 都完成,并收到两份结果
    assert "任务 1: 2" in user_input, user_input
    assert "任务 2: 4" in user_input, user_input
    return "总和是 6"


def fake_create_par(**kwargs):
    msgs = kwargs.get("messages", [])
    if msgs[0]["content"] == multiagent.PLANNER_SYSTEM:
        has_results = any("已完成子任务的结果" in m.get("content", "") for m in msgs[1:])
        if has_results:
            return FakeResp("完成,总和是 6。")
        return FakeResp(
            '{"tasks": [{"task_id": 1, "agent": "calculator", "description": "计算 1+1", '
            '"reason": "r"}, {"task_id": 2, "agent": "calculator", "description": "计算 2+2", '
            '"reason": "r"}, {"task_id": 3, "agent": "general", '
            '"description": "汇总任务1和任务2的结果", "reason": "依赖前两者", '
            '"depends_on": [1, 2]}]}'
        )
    return FakeResp("")


with patch.object(multiagent.agent.client.chat.completions, "create", fake_create_par), \
     patch("multiagent.agent.run_agent", side_effect=fake_executor_par):
    out = run_multiagent("并行测试", max_steps=4)

t1s = next(e[2] for e in par_events if e[:2] == ("t1", "start"))
t1e = next(e[2] for e in par_events if e[:2] == ("t1", "end"))
t2s = next(e[2] for e in par_events if e[:2] == ("t2", "start"))
t2e = next(e[2] for e in par_events if e[:2] == ("t2", "end"))
# 两个独立任务时间窗重叠 = 真正并行(若串行,t2 会在 t1 结束后才开始)
assert t2s < t1e and t1s < t2e, f"独立任务应并行执行: t1[{t1s:.2f},{t1e:.2f}] t2[{t2s:.2f},{t2e:.2f}]"
assert "6" in out
print("OK  并行:两个独立子任务时间窗重叠(真并行),依赖任务收到两份前置结果")
print("    t1 窗口 [%.2f, %.2f] | t2 窗口 [%.2f, %.2f]" % (t1s, t1e, t2s, t2e))


# --- 测试 12:depends_on 引用不存在 task_id -> 死锁兜底,不阻塞整批 ---
stuck_out = []


def fake_executor_stuck(user_input, **kwargs):
    stuck_out.append(user_input)
    return "3"


def fake_create_stuck(**kwargs):
    msgs = kwargs.get("messages", [])
    if msgs[0]["content"] == multiagent.PLANNER_SYSTEM:
        has_results = any("已完成子任务的结果" in m.get("content", "") for m in msgs[1:])
        if has_results:
            return FakeResp("结果是 3。")
        return FakeResp(
            '{"tasks": [{"task_id": 1, "agent": "calculator", "description": "算 1+2", '
            '"reason": "r", "depends_on": [99]}]}'
        )
    return FakeResp("")


with patch.object(multiagent.agent.client.chat.completions, "create", fake_create_stuck), \
     patch("multiagent.agent.run_agent", side_effect=fake_executor_stuck):
    out = run_multiagent("死锁测试", max_steps=4)

assert len(stuck_out) == 1, f"依赖不存在的任务应降级执行而非卡死: {len(stuck_out)} 次"
assert "3" in out
print("OK  死锁兜底:depends_on 引用不存在的 task_id -> 任务降级执行,不阻塞")


print("\n全部通过")
