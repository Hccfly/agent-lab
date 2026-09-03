"""命令行入口:支持会话记忆 + 多 agent 协作 + 流式输出,带 /new /list /resume /state /multi 命令。

用法示例:
    现在几点?                       -> 单 agent,调用 get_current_time(流式输出)
    /multi 帮我查一下明天是几号     -> 多 agent(Planner 拆解 + Executor 执行)
    /new                            -> 开启新会话
    /list                           -> 列出所有会话
    /resume <id>                    -> 继续旧会话
    /state                          -> 查看当前会话的摘要和历史
"""
import datetime
import sys

from agent import run_agent
from log import log_event, setup_logging
from memory import list_sessions, load_session
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


def main():
    setup_logging()
    session_id = new_session_id()
    log_event("session_start", f"启动会话 {session_id}", session_id=session_id)
    show_reasoning = True  # 合流模式:显示模型的 Thought(推理),/thinking 切换
    print("最小 Agent 演示(支持会话记忆 + 多 agent + 流式,输入 exit 退出)")
    print(f"当前会话: {session_id}")
    print("命令: /new 新会话 | /list 列出会话 | /resume <id> 继续旧会话 | /state 查看状态 | /multi <目标> 多agent协作(专用角色分工·独立子任务并行) | /react <问题> ReAct显式推理 | /thinking 切换Thought显示")

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
            session_id = sid
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
