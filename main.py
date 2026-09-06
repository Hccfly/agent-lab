"""命令行入口:支持手写与 LangGraph 两种 Agent 编排方式。

用法示例:
    现在几点?                       -> 单 agent,调用 get_current_time(流式输出)
    /multi 帮我查一下明天是几号     -> 多 agent(Planner 拆解 + Executor 执行)
    /graph 帮我计算 (12+8)*3         -> LangGraph 显式状态图版单 agent
    /graph-multi 完成复合任务         -> LangGraph DAG 并行多 Agent
    /checkpoint-start demo | 完成任务  -> SQLite checkpoint 启动并在执行波次后暂停
    /checkpoint-resume demo           -> 使用相同 thread_id 从断点继续
    /hitl-start review | 读取文件      -> 敏感工具执行前暂停并等待审批
    /hitl-approve review              -> 批准当前敏感工具调用
    /new                            -> 开启新会话
    /list                           -> 列出所有会话
    /resume <id>                    -> 继续旧会话
    /state                          -> 查看当前会话的摘要和历史
"""
import datetime
import json
import sys

from agent import run_agent
from checkpointing import CheckpointRun, CheckpointThreadError, PersistentMultiAgent
from hitl import ApprovalRun, ApprovalThreadError, PersistentApprovalAgent
from langgraph_agent import invoke_agent
from langgraph_multiagent import invoke_multiagent
from log import log_event, setup_logging
from memory import list_sessions, load_session, validate_session_id
from multiagent import run_multiagent
from react import run_react


def new_session_id() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def _stream_out(text: str) -> None:
    """流式输出的落地:把文本写到 stdout 并立刻刷新。"""
    sys.stdout.write(text)
    sys.stdout.flush()


def _reasoning_out(text: str) -> None:
    """合流模式:把模型的 Thought(推理)以灰色写到 stdout。

    ANSI 灰色码 \033[90m ... \033[0m;Windows 终端一般支持,不支持时
    退化为普通文本,不影响功能。
    """
    sys.stdout.write(f"\033[90m{text}\033[0m")
    sys.stdout.flush()


def _print_checkpoint_run(run: CheckpointRun) -> None:
    state = run.state
    status = "已完成" if run.completed else "已暂停"
    print(f"[checkpoint] thread={run.thread_id} | {status} | next={list(run.next_nodes)}")
    print(
        f"[checkpoint] 成功={len(state.get('task_results', {}))}/"
        f"{len(state.get('tasks', []))} | 波次={state.get('wave', 0)} | "
        f"checkpoint_id={run.checkpoint_id}"
    )
    if run.completed and state.get("answer"):
        print(state["answer"])


def _print_approval_run(run: ApprovalRun) -> None:
    if run.waiting_for_approval:
        print(f"[hitl] thread={run.thread_id} | 等待审批 | next={list(run.next_nodes)}")
        for call in run.pending_calls:
            print(f"  call_id={call['id']} | tool={call['name']} | arguments={call['arguments']}")
        print("可选择 /hitl-approve、/hitl-reject 或 /hitl-edit")
    elif run.completed:
        print(f"[hitl] thread={run.thread_id} | 已完成")
        if run.state.get("answer"):
            print(run.state["answer"])
    else:
        print(f"[hitl] thread={run.thread_id} | 运行中 | next={list(run.next_nodes)}")


def main():
    setup_logging()
    session_id = new_session_id()
    log_event("session_start", f"启动会话 {session_id}", session_id=session_id)
    show_reasoning = True  # 合流模式:显示模型的 Thought(推理),/thinking 切换
    print("最小 Agent 演示(支持会话记忆 + 多 agent + 流式,输入 exit 退出)")
    print(f"当前会话: {session_id}")
    print("命令: /new 新会话 | /list 列出会话 | /resume <id> 继续旧会话 | /state 查看状态 | /multi <目标> 手写多agent | /graph-multi <目标> LangGraph多agent | /checkpoint-start <thread_id> | <目标> | /checkpoint-resume <thread_id> | /hitl-start <thread_id> | <问题> | /hitl-approve <thread_id> | /hitl-reject <thread_id> | <原因> | /hitl-edit <thread_id> | <JSON参数> | /graph <问题> LangGraph单agent")

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        if user_input == "/new":
            session_id = new_session_id()
            log_event("session_start", f"新会话开始: {session_id}", session_id=session_id)
            print(f"新会话开始: {session_id}")
            continue
        if user_input == "/list":
            sessions = list_sessions()
            if not sessions:
                print("还没有会话")
            else:
                for sid, n in sessions:
                    print(f"  {sid}  ({n} 条消息)")
            continue
        if user_input.startswith("/resume"):
            sid = user_input.removeprefix("/resume").strip()
            if not sid:
                sid = input("会话 ID: ").strip()
            try:
                session_id = validate_session_id(sid)
            except ValueError as exc:
                print(f"会话 ID 无效: {exc}")
                continue
            log_event("session_resume", f"已切换到会话: {session_id}", session_id=session_id)
            print(f"已切换到会话: {session_id}")
            continue
        if user_input.startswith("/multi"):
            goal = user_input.removeprefix("/multi").strip()
            if not goal:
                print("用法: /multi <目标>,例如 /multi 帮我算一下明天是几号")
                continue
            log_event("user_input", "", session_id=session_id, mode="multi", input=goal[:200])
            print("--- 多 agent 协作开始 ---")
            print(run_multiagent(goal, stream=True, on_delta=_stream_out))
            print("--- 多 agent 协作结束 ---")
            continue
        if user_input.startswith("/hitl-start"):
            payload = user_input.removeprefix("/hitl-start").strip()
            if "|" not in payload:
                print("用法: /hitl-start <thread_id> | <问题>")
                continue
            thread_id, question = (part.strip() for part in payload.split("|", 1))
            try:
                with PersistentApprovalAgent() as agent:
                    run = agent.start(question, thread_id)
                _print_approval_run(run)
            except (ApprovalThreadError, ValueError) as exc:
                print(f"HITL 启动失败: {exc}")
            continue
        if user_input.startswith("/hitl-status"):
            thread_id = user_input.removeprefix("/hitl-status").strip()
            if not thread_id:
                print("用法: /hitl-status <thread_id>")
                continue
            try:
                with PersistentApprovalAgent() as agent:
                    run = agent.status(thread_id)
                _print_approval_run(run)
            except (ApprovalThreadError, ValueError) as exc:
                print(f"HITL 查询失败: {exc}")
            continue
        if user_input.startswith("/hitl-approve"):
            thread_id = user_input.removeprefix("/hitl-approve").strip()
            if not thread_id:
                print("用法: /hitl-approve <thread_id>")
                continue
            try:
                with PersistentApprovalAgent() as agent:
                    run = agent.approve(thread_id)
                _print_approval_run(run)
            except (ApprovalThreadError, ValueError) as exc:
                print(f"HITL 审批失败: {exc}")
            continue
        if user_input.startswith("/hitl-reject"):
            payload = user_input.removeprefix("/hitl-reject").strip()
            if "|" not in payload:
                print("用法: /hitl-reject <thread_id> | <拒绝原因>")
                continue
            thread_id, reason = (part.strip() for part in payload.split("|", 1))
            try:
                with PersistentApprovalAgent() as agent:
                    run = agent.reject(thread_id, reason)
                _print_approval_run(run)
            except (ApprovalThreadError, ValueError) as exc:
                print(f"HITL 拒绝失败: {exc}")
            continue
        if user_input.startswith("/hitl-edit"):
            payload = user_input.removeprefix("/hitl-edit").strip()
            if "|" not in payload:
                print('用法: /hitl-edit <thread_id> | {"path":"新路径"}')
                continue
            thread_id, raw_arguments = (part.strip() for part in payload.split("|", 1))
            try:
                arguments = json.loads(raw_arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("修改后的参数必须是 JSON 对象")
                with PersistentApprovalAgent() as agent:
                    run = agent.edit(thread_id, arguments)
                _print_approval_run(run)
            except (ApprovalThreadError, ValueError) as exc:
                print(f"HITL 修改失败: {exc}")
            continue
        if user_input.startswith("/checkpoint-start"):
            payload = user_input.removeprefix("/checkpoint-start").strip()
            if "|" not in payload:
                print("用法: /checkpoint-start <thread_id> | <目标>")
                continue
            thread_id, goal = (part.strip() for part in payload.split("|", 1))
            try:
                with PersistentMultiAgent(interrupt_after=["executor"]) as runner:
                    run = runner.start(goal, thread_id)
                _print_checkpoint_run(run)
            except (CheckpointThreadError, ValueError) as exc:
                print(f"checkpoint 启动失败: {exc}")
            continue
        if user_input.startswith("/checkpoint-resume"):
            thread_id = user_input.removeprefix("/checkpoint-resume").strip()
            if not thread_id:
                print("用法: /checkpoint-resume <thread_id>")
                continue
            try:
                with PersistentMultiAgent(interrupt_after=["executor"]) as runner:
                    run = runner.resume(thread_id)
                _print_checkpoint_run(run)
            except (CheckpointThreadError, ValueError) as exc:
                print(f"checkpoint 恢复失败: {exc}")
            continue
        if user_input.startswith("/checkpoint-status"):
            thread_id = user_input.removeprefix("/checkpoint-status").strip()
            if not thread_id:
                print("用法: /checkpoint-status <thread_id>")
                continue
            try:
                with PersistentMultiAgent(interrupt_after=["executor"]) as runner:
                    run = runner.status(thread_id)
                _print_checkpoint_run(run)
            except (CheckpointThreadError, ValueError) as exc:
                print(f"checkpoint 查询失败: {exc}")
            continue
        if user_input.startswith("/graph-multi"):
            goal = user_input.removeprefix("/graph-multi").strip()
            if not goal:
                print("用法: /graph-multi <目标>,例如 /graph-multi 计算 12*5 并查询当前时间")
                continue
            log_event("user_input", "", session_id=session_id, mode="langgraph_multi", input=goal[:200])
            state = invoke_multiagent(goal)
            print("--- LangGraph 多 Agent 模式 ---")
            print(state["answer"])
            # trace 是合并后的事件列表，不代表并行分支的实际先后顺序。
            for event in state["trace"]:
                if event["node"] == "scheduler" and event.get("task_ids"):
                    print(f"[wave {event['wave']}] 就绪任务: {event['task_ids']}（最多 4 个并发）")
            for tid, error in sorted(state["task_errors"].items()):
                print(f"[task {tid}] {state['task_status'][tid]}: {error}")
            print(f"[graph-multi] 成功={len(state['task_results'])}/{len(state['tasks'])} | 波次={state['wave']} | stop={state['stop_reason']}")
            continue
        if user_input.startswith("/graph"):
            question = user_input.removeprefix("/graph").strip()
            if not question:
                print("用法: /graph <问题>,例如 /graph 帮我计算 (12+8)*3")
                continue
            log_event("user_input", "", session_id=session_id, mode="langgraph", input=question[:200])
            state = invoke_agent(question)
            print("--- LangGraph 状态图模式 ---")
            print(state["answer"])
            path = " -> ".join(event["node"] for event in state["trace"])
            print(f"[graph] {path} | 模型轮次={state['iteration']} | stop={state['stop_reason']}")
            continue
        if user_input == "/thinking":
            show_reasoning = not show_reasoning
            print(f"Thought 显示: {'开' if show_reasoning else '关'}")
            continue
        if user_input.startswith("/react"):
            question = user_input.removeprefix("/react").strip()
            if not question:
                print("用法: /react <问题>,例如 /react 帮我算一下 3 加 5 乘 2")
                continue
            log_event("user_input", "", session_id=session_id, mode="react", input=question[:200])
            print("--- ReAct 模式(Thought / Action / Observation) ---")
            print(run_react(question, stream=True, on_delta=_stream_out))
            print("--- ReAct 结束 ---")
            continue
        if user_input == "/state":
            st = load_session(session_id)
            print(f"会话: {session_id}")
            print(f"摘要: {st['summary'] or '(无)'}")
            print(f"历史消息数: {len(st['messages'])}")
            continue
        if not user_input:
            continue

        log_event("user_input", "", session_id=session_id, mode="single", input=user_input[:200])
        print("Agent: ", end="", flush=True)
        run_agent(
            user_input,
            session_id=session_id,
            stream=True,
            on_delta=_stream_out,
            show_reasoning=show_reasoning,
            on_reasoning=_reasoning_out if show_reasoning else None,
        )
        print()  # 换行,结束流式输出


if __name__ == "__main__":
    main()
