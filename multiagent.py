"""多 agent 协作:Planner + 多专用 Executor + Orchestrator。

设计要点:
- 为什么拆成两级? 单 agent 面对复杂任务容易"贪多"漏细节;Planner 拆解后
  交给专注的 Executor,每个子任务上下文更聚焦、更可靠。
- 为什么是"多专用"Executor? 每个角色只持有自己该有的工具子集——
  能力最小化(最小权限)+ 上下文聚焦。文件 agent 拿不到 calculator,
  天然不能越权计算;模型看到的工具少,也不幻想着调用能力之外的工具。
- Planner 输出结构化 JSON(而不是自由文本):可被 Orchestrator 稳定解析、
  可校验、可审计;自由文本拆解不可靠。
- Orchestrator 是协调者:维护"子任务结果表",按任务归属(agent 字段)
  路由到专用 Executor,把结果回填给 Planner 驱动它迭代。

流程:
    Planner(拆解任务 -> JSON,每任务标注 agent 归属 + 可选的 depends_on 依赖)
        -> Orchestrator 按 depends_on 拓扑分层(依赖层间串行)
        -> 无依赖的同一层任务并行执行(ThreadPoolExecutor)
        -> Executor 用角色自己的工具子集执行子任务
        -> Orchestrator 把结果回填给 Planner
        -> Planner 判断:完成 -> 输出最终答案 / 未完成 -> 再拆或再问
        -> 直到产出最终答案或达到 max_steps

并行与依赖(核心设计):
- Planner 拆出的每个任务可声明 depends_on(它依赖的前置 task_id 列表)。
  Orchestrator 不猜任务关系,只信 Planner 的声明 + 已完成结果表。
- 本批任务按 depends_on 拓扑分层:无依赖(或依赖已完成)的任务进同一波,
  波内并行;依赖本波其他任务的后置任务进后面的波,等前置完成再执行。
- 依赖已完成的同层并行加速是"多 agent 并行"的第二个卖点;
  与依赖分批(跨批串行)配合:有数据依赖的跨批由 Planner 分批 + 结果回填,
  批内并行只作用于真正独立的子任务,绝不破坏依赖语义。
- stream=True 时降级串行:流式输出是单通道回调,并发写会交织;
  并行是性能特性、流式是展示特性,交互模式下牺牲并行保证可读。
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor

import agent  # 复用 run_agent(工具循环 + 记忆)
from log import log_event
from tools import TOOLS, tool_subset

PLANNER_SYSTEM = """你是一个任务规划者。用户会给你一个复杂目标,你要把它拆解成
可执行的子任务清单。

要求:
1. 输出严格的 JSON,不要任何多余文字、不要 markdown 代码块标记。
2. JSON 结构固定为:
   {"tasks": [{"task_id": 1, "agent": "calculator", "description": "...", "reason": "..."}]}
   每个任务可带可选字段 depends_on:该任务依赖的前置任务 task_id 数组(无依赖可省略)。
3. agent 是执行该任务的专用角色,只能取以下之一:
   - general(通用,可调全部工具)
   - calculator(计算,只会做数学计算)
   - time(时间,只会查当前日期时间)
   - file(文件,只会读文件内容)
4. 按任务内容挑最合适的 agent;拿不准就用 general。
5. description 是给执行者的具体指令(用简体中文),reason 是拆解理由。
6. 只输出需要执行的任务;如果无需拆解、可直接回答,输出 {"tasks": []}。
7. 子任务数量一般不超过 5 个,能合并就合并。
8. **依赖分批与并行(关键)**:子任务的数据依赖用两种方式表达,不要混:
   - 跨批(最稳):如果某个子任务需要"上一个子任务执行后才知道"的内容
     (比如先读文件、再统计文件内容),不要把这种有依赖的任务放在同一批拆解。
     先只拆当前能独立执行的;等收到已完成子任务的结果后,下一轮再拆依赖它们
     的新任务(此时描述里可以直接引用拿到手的具体内容)。
   - 同批 + depends_on:若你确实要把有依赖的任务放同一批,必须给后置任务加
     "depends_on": [前置task_id];Orchestrator 会等前置任务完成后才执行它。
   - 能独立执行的任务(无数据依赖)放同一批即可——Orchestrator 会并行执行
     它们来加速;不要把会互相引用结果的任务当成独立任务并行。

在执行过程中,你还会收到"已完成子任务的结果"。基于这些结果:
- 如果所有子任务都完成了,输出最终答案(可以是自由文本,不再是 JSON)。
- 如果还需要补充执行,输出新的 JSON 拆解。"""

_ROLE_NAMES = ("general", "calculator", "time", "file")


def _role_system(label: str, tool_names: str) -> str:
    """构造专用 executor 的系统提示:强调角色身份与能力边界。

    设计要点:专用 agent 的价值一半在"知道自己不会什么"。system prompt
    明说能力边界,模型遇到能力之外的任务就明确拒绝,而不是硬编造。
    """
    return (
        f"你是{label}专员 agent。作为多 agent 协作中的执行者,你只负责"
        f"自己分工内的子任务。\n\n"
        f"你能使用的工具(只能从这些里选):{tool_names}。\n"
        "判断是否需要工具:任务需要精确计算/实时信息/读文件时,必须调用"
        "对应工具,不要编造结果。\n"
        "如果子任务需要你能力之外的步骤(需要其他角色的工具),明确说出"
        "你无法完成该部分,不要假装完成。\n"
        "回答使用简体中文,直接返回这个子任务的结果。"
    )


# 声明式角色注册表:新增一个专用角色只需加一条。
# general 兜底全量工具,保证 Planner 标错/漏标时任务仍能执行。
ROLE_REGISTRY: dict[str, dict] = {
    "general": {
        "label": "通用",
        "tools": TOOLS,
        "system": _role_system("通用", "get_current_time, calculator, read_file"),
    },
    "calculator": {
        "label": "计算",
        "tools": tool_subset("calculator"),
        "system": _role_system("计算", "calculator"),
    },
    "time": {
        "label": "时间",
        "tools": tool_subset("get_current_time"),
        "system": _role_system("时间", "get_current_time"),
    },
    "file": {
        "label": "文件",
        "tools": tool_subset("read_file"),
        "system": _role_system("文件", "read_file"),
    },
}

def _planner_start(goal: str) -> str:
    return (
        f"请拆解下面的目标为子任务清单:\n\n目标: {goal}\n\n"
        '仅输出 JSON,格式: {"tasks": [{"task_id": 1, "agent": "general", '
        '"description": "...", "reason": "..."}]}\n'
        "depends_on 可选:任务依赖的前置 task_id 数组,如 \"depends_on\": [1];无依赖省略。\n"
        f"agent 只能取: {'/'.join(_ROLE_NAMES)}"
    )

EXECUTOR_TEMPLATE = (
    "执行这个子任务并用你分内的工具完成它。\n"
    "子任务: {description}\n"
    "如果子任务需要你分内的能力(计算/时间/读文件),必须调用对应工具,不要编造结果。"
    "直接返回这个子任务的结果。"
)

# Executor 结果命中这些信号 = 子任务没完成。常见根因两类:
# ① 角色能力不足(Planner 把"读文件"派给了无 read_file 的 calculator 角色,
#    角色 system 教它"明确说出你无法完成",于是返回这类文本);
# ② 缺前置上下文(Planner 仍把依赖任务同批拆,后置 Executor 拿不到数据)。
# 命中后 Orchestrator 会用全量工具的 general 重派一次(见 RETRY_TEMPLATE)。
# 召回优先:宁可对误报多花一次重派,也不放过一个没完成的任务。
EXECUTOR_FAIL_SIGNALS = (
    "无法完成", "不能完成", "无法执行", "无法处理", "无能为力",
    "超出我的能力", "超出能力范围", "能力之外", "没有能力", "超出我的权限",
    "缺少", "缺失", "没有提供", "未提供", "没有收到",
    "请提供", "需要提供", "需要文件路径", "需要文件内容", "需要路径", "需要上下文",
    "无法读取", "读不到", "没有文件", "文件不存在", "无法访问",
)

RETRY_TEMPLATE = (
    "子任务: {description}\n\n"
    "你上一次执行返回了无法完成的信息:\n"
    "> {failure}\n\n"
    "现在允许你使用全部工具(get_current_time / calculator / read_file)重新完成它。\n"
    "本次多 agent 协作中其他已完成任务的结果就在下面,需要时直接使用:\n"
    "{context}\n"
    "如果这个子任务明确要求读取某个文件,直接用 read_file 工具读取内容"
    "(从上面的结果里或你的判断中找出文件路径),再基于读到的内容完成,\n"
    "不要因为缺信息就放弃。再次无法完成就简明说明原因。"
)


def _is_executor_failure(text: str) -> bool:
    """粗略判断一次子任务执行是否以"无法完成"告终(命中失败信号)。"""
    t = (text or "").strip()
    if not t:
        return False
    return any(s in t for s in EXECUTOR_FAIL_SIGNALS)


# 每波并行上限:防止并发 LLM 调用过多撞服务限流。
MAX_PARALLEL = 4


def _topological_waves(tasks: list[dict], done: set[int]) -> list[list[dict]]:
    """把本批 tasks 按 depends_on 拓扑分成若干波;同一波内的任务互不依赖,可并行。

    done: 已完成的 task_id 集合(此前轮次 + 本批更早的波)。
    - 依赖在本批内更晚才完成的任务,会被推到后面的波,等前置完成再执行;
    - 依赖已满足(无 depends_on,或 depends_on 全在 done)的任务进当前波。
    - 死锁兜底:若某任务 depends_on 引用了本批和 done 都不存在的 task_id
      (Planner 幻觉),依赖永远无法满足——把它降级为逐个串行尽力执行,
      不阻塞整批(失败自愈会兜执行失败)。
    """
    remaining = {t["task_id"]: t for t in tasks}
    deps = {t["task_id"]: set(t.get("depends_on") or []) for t in tasks}
    waves: list[list[dict]] = []
    while remaining:
        ready = [t for tid, t in remaining.items() if deps[tid] <= done]
        if not ready:
            stuck = list(remaining.values())
            log_event(
                "planner_dep_stuck",
                f"[orchestrator] {len(stuck)} 个任务依赖引用不存在的 task_id,降级逐个串行",
                stuck=sorted(t["task_id"] for t in stuck),
            )
            for t in stuck:  # 逐个成波 = 串行,避免伪并行下互相依赖的竞态
                waves.append([t])
            return waves
        for t in ready:
            remaining.pop(t["task_id"])
        waves.append(ready)
        done.update(t["task_id"] for t in ready)
    return waves


def _execute_one(task: dict, prev_results: dict, *, stream: bool, on_delta) -> tuple[int, str]:
    """执行单个子任务:路由 -> 注入前置结果 -> run_agent -> 失败自愈重派。

    prev_results: 本任务分派时已完成的子任务结果快照——只含它真正可见的
    前置结果。并行波内任务互不依赖(拓扑保证),所以不引用彼此结果;
    跨波/跨批的依赖任务,执行时前置已在 prev_results 里,注入即可。

    返回 (task_id, 最终结果),由调用方写入结果表。
    """
    tid = task["task_id"]
    description = task["description"]
    role = task.get("agent") or "general"
    cfg = ROLE_REGISTRY.get(role) or ROLE_REGISTRY["general"]
    if role not in ROLE_REGISTRY:
        log_event(
            "planner_role_fallback", f"[planner] 未知角色 {role!r},兜底 general",
            task_id=tid, role=role,
        )
    log_event(
        "planner_decompose",
        f"[planner] 子任务 {tid} -> {cfg['label']}agent: {description[:60]}",
        task_id=tid, role=role, description=description, depends_on=task.get("depends_on"),
    )
    # 把已完成的子任务结果附给当前任务作为参考上下文:
    # 依赖任务(依赖前波/前批)能从真实结果里取,而不是假设自己已知。
    exec_prompt = EXECUTOR_TEMPLATE.format(description=description)
    if prev_results:
        prev = "\n".join(f"任务 {t}: {r}" for t, r in sorted(prev_results.items()))
        exec_prompt += (
            "\n\n重要: 以下内容是本次多 agent 协作中已完成的前置任务的真实结果。"
            "如果你的子任务描述提到要分析'读取到/获取到/上一步的'内容(例如一段文本、"
            "一份文件的内容),你需要的内容已经在下面这些结果里。直接基于这些内容完成"
            "子任务,不要尝试自己读取文件,也不要索要文件路径:\n"
            + prev[:800]
        )
        log_event(
            "executor_context", f"[orchestrator] 已附 {len(prev_results)} 条前置结果给任务 {tid}",
            task_id=tid, prev_results=len(prev_results),
        )
    result = agent.run_agent(
        exec_prompt,
        session_id=None,
        stream=stream,
        on_delta=on_delta,
        tools=cfg["tools"],
        system_prompt=cfg["system"],
    )

    # 失败自愈(硬兜底):Executor 明确表示无法完成(能力不足 / 缺上下文)。
    # 自动用全量工具的 general 重派一次,把失败原因 + 全部前置结果带回去。
    # 这是对"Planner 提示分批 + Orchestrator 附前置结果"两道软约束之上的
    # 第三道防线——模型不听话时,系统自己兜住,而不是让 Planner 吞下失败。
    # 只重派一次:仍失败就把失败信息回填给 Planner,由它决定下一步。
    if _is_executor_failure(result):
        context = "\n".join(
            f"任务 {t}: {r}" for t, r in sorted(prev_results.items())
        )
        retry_prompt = (
            RETRY_TEMPLATE
            .replace("{description}", description)
            .replace("{failure}", result[:600])
            .replace("{context}", context or "(本轮暂无其他已完成结果)")
        )
        log_event(
            "executor_retry",
            f"[orchestrator] 任务 {tid} 执行失败,自动用通用角色重派一次",
            task_id=tid, role=role, failure=result[:200],
        )
        result = agent.run_agent(
            retry_prompt,
            session_id=None,
            stream=stream,
            on_delta=on_delta,
            tools=ROLE_REGISTRY["general"]["tools"],
            system_prompt=ROLE_REGISTRY["general"]["system"],
        )
        log_event(
            "executor_retry_result",
            f"    -> [通用agent] 重派结果: {result[:80]}",
            task_id=tid, retried=True, ok=not _is_executor_failure(result),
        )
    log_event(
        "executor_result",
        f"    -> [{cfg['label']}agent] 结果: {result[:80]}",
        task_id=tid, role=role, result=result[:200],
    )
    return tid, result


def _strip_code_fence(text: str) -> str:
    """去掉 ```json ... ``` 代码块包裹(容忍模型输出 markdown 标记)。"""
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    return m.group(1) if m else text


def _extract_json(text: str) -> dict:
    """从模型输出里稳健地提取 JSON。容忍可能的代码块或前后杂音。"""
    text = _strip_code_fence(text).strip()
    # 找到第一个 { 到最后一个 } 之间
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"模型输出中没有 JSON: {text[:100]}")
    return json.loads(text[start : end + 1])


def run_multiagent(
    goal: str,
    max_steps: int = 8,
    stream: bool = False,
    on_delta=None,
) -> str:
    """Orchestrator:驱动 Planner 拆解 -> Executor 执行 -> 回填 -> 收敛。

    注意:整个多 agent 流程刻意不使用会话记忆——
    Planner 的拆解和 Executor 的子任务都是"临时工作记忆",
    不该混入长期记忆(否则会污染摘要和 RAG 索引)。
    最终答案由调用方决定是否持久化。

    stream=True 时,Executor 的子任务执行与回答会逐字回调 on_delta;
    Planner 的输出是结构化 JSON(控制面),不适合流式,保持一次性返回。
    """
    planner_msgs = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": _planner_start(goal)},
    ]
    task_results: dict[int, str] = {}  # 子任务结果表

    for step in range(max_steps):
        # 把已完成子任务的结果回填给 Planner,让它判断下一步
        if task_results:
            filled = "\n".join(
                f"任务 {tid}: {res}" for tid, res in sorted(task_results.items())
            )
            planner_msgs.append(
                {
                    "role": "user",
                    "content": f"已完成子任务的结果:\n{filled}\n\n"
                    "如果这些结果足以回答目标,请直接输出最终答案(自由文本);"
                    "否则继续输出需要补充的子任务 JSON。",
                }
            )

        resp = agent.client.chat.completions.create(
            model=agent.MODEL,
            messages=planner_msgs,
        )
        planner_text = resp.choices[0].message.content or ""
        planner_msgs.append({"role": "assistant", "content": planner_text})
        log_event("planner_output", "", step=step, chars=len(planner_text), is_json=_strip_code_fence(planner_text).strip().startswith("{"))

        # 判断 Planner 是输出了最终答案(非 JSON)还是拆解(JSON)。
        # 注意:模型偶尔会无视"不要 markdown 代码块"的指令,用 ```json 包裹 JSON。
        # 先剥离代码块再判断,否则 ```json {...} 会被误判成"最终答案"原样返回。
        stripped = _strip_code_fence(planner_text).strip()
        if not stripped.startswith("{"):
            # 非 JSON -> 最终答案
            return planner_text.strip()

        try:
            plan = _extract_json(planner_text)
        except Exception:
            # 拆解失败:退化为直接把目标交给 Executor 兜底,避免死循环。
            # 不传 session_id:兜底执行是临时工作,不写入会话记忆。
            result = agent.run_agent(goal, stream=stream, on_delta=on_delta)
            return f"[拆解失败,已直接执行]\n{result}"

        tasks = plan.get("tasks", [])
        if not tasks:
            # 空拆解 = 无需子任务,让它直接回答
            return "Planner 未拆解出子任务,没有可执行内容。"

        # 按 depends_on 拓扑分层:同波任务互不依赖 -> 并行;层间串行。
        # 刻意不传 session_id:子任务执行是"临时工作记忆",不该混入长期记忆,
        # 否则"子任务描述"会污染会话历史、摘要和 RAG 索引。
        waves = _topological_waves(tasks, set(task_results))
        for wave in waves:
            # 快照 = 本波开始前已完成的结果。依赖任务在更晚的波,执行时
            # 前置结果已在快照里;并行波内任务互不依赖,不需要看彼此。
            snapshot = dict(task_results)
            if len(wave) > 1 and not stream:
                # 并行分支:批内真正独立的子任务并发执行,加速多 agent。
                # 仅在 stream=False 时启用——流式回调 on_delta 是单通道,
                # 并发写会交织(并行是性能特性,流式是展示特性,二选一)。
                workers = min(len(wave), MAX_PARALLEL)
                log_event(
                    "executor_parallel",
                    f"[orchestrator] 本波 {len(wave)} 个独立子任务并行执行(workers={workers})",
                    wave=[t["task_id"] for t in wave], workers=workers,
                )
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [
                        pool.submit(_execute_one, t, snapshot, stream=False, on_delta=None)
                        for t in wave
                    ]
                    for fut in futures:
                        tid, result = fut.result()
                        task_results[tid] = result
            else:
                # 串行分支:单任务波,或 stream=True 需保持输出顺序。
                for t in wave:
                    tid, result = _execute_one(
                        t, snapshot, stream=stream, on_delta=on_delta
                    )
                    task_results[tid] = result

    # 兜底:达到 max_steps 仍未收敛
    return f"达到最大规划步数({max_steps}),未完全收敛。\n已收集的子任务结果:\n" + "\n".join(
        f"任务 {tid}: {res}" for tid, res in sorted(task_results.items())
    )
