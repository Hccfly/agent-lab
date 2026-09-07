"""ReAct 显式化:Thought/Action/Observation/Final Answer 四元组循环。

两种 agent 范式的对比:
- 原生 tool calling(agent.py):模型直接返回结构化 tool_calls,SDK 保证格式,
  agent 循环只管执行并回填。格式可靠,但"为什么调用"是隐式的。
- ReAct 文本协议(react.py):模型自由输出文本,agent 用解析器提取 Action。
  Thought 先于 Action 显式可见,推理链可解释、可审计;代价是要自己解析,
  模型格式容易漂移(需要鲁棒解析器 + 兜底)。

ReAct 论文(Yao et al. 2022)的核心:
    Thought(推理) -> Action(工具) -> Observation(观察) -> 循环
    -> ... -> Final Answer

本实现复用现有分层:tools.run_tool 执行工具、log.log_event 结构化日志、
tools 的幂等去重(tool_key + IDEMPOTENT_TOOLS)同样生效。
"""
import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

from log import log_event
from tools import IDEMPOTENT_TOOLS, run_tool, tool_key
from governance import env_positive_float

client = OpenAI()  # 复用 agent.py 同一套环境变量
MODEL = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")
MODEL_TIMEOUT_SECONDS = env_positive_float("AGENT_MODEL_TIMEOUT_SECONDS", 30.0)

REACT_SYSTEM = """你是一个使用 ReAct(Reasoning + Acting)模式的助手。处理问题时遵循这个循环:

Thought: 用一两句话推理——你已知什么、还差什么、下一步该做什么、为什么。
Action: 要调用的工具名(只能从给定的工具里选)。
Action Input: 给该工具的参数,必须是合法 JSON(一行或多行均可)。

你会收到 Observation: <工具结果>,然后继续 Thought -> Action -> ...
当你不再需要任何工具、可以回答用户时,直接输出:

Final Answer: <给用户的最终回答>

注意:
- Thought 不能省略,先想清楚再行动。
- Action 只能调用以下工具之一,参数必须匹配:
  - get_current_time(无参数,直接写 {})
  - calculator,参数 {"expression": "数学表达式"}
  - read_file,参数 {"path": "文件绝对路径"}
- 不要编造工具的返回结果,只基于 Observation 组织回答。
- 回答使用简体中文。"""


def _call_model(messages: list[dict], stream: bool, on_delta=None) -> str:
    """调模型返回完整文本。ReAct 是纯文本协议,不传 tools。

    流式时把内容逐字回调 on_delta(打字机效果),同时累积完整文本供解析。
    """
    if stream:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            timeout=MODEL_TIMEOUT_SECONDS,
        )
        parts: list[str] = []
        usage = None
        for chunk in resp:
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                parts.append(delta.content)
                if on_delta:
                    on_delta(delta.content)
        if usage is not None:
            log_event("llm_call", f"[llm] 本轮 tokens={usage.total_tokens}", tokens=usage.total_tokens)
        return "".join(parts)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        timeout=MODEL_TIMEOUT_SECONDS,
    )
    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    if usage is not None:
        log_event("llm_call", f"[llm] 本轮 tokens={usage.total_tokens}", tokens=usage.total_tokens)
    return text


# Action 行:容忍 "Action 1:" 这类带编号的格式(论文常见)。
_ACTION_RE = re.compile(
    r"Action\s*\d*\s*:\s*(?P<name>[\w_]+)\s*\n\s*Action\s+Input\s*\d*\s*:\s*(?P<input>\{.*?\})",
    re.DOTALL,
)
# Final Answer 行:可能跨多行。
_FINAL_RE = re.compile(r"Final\s+Answer\s*:\s*(?P<final>[\s\S]*)")


def _parse(text: str) -> tuple:
    """解析模型的一轮输出。返回 (kind, payload):
    - ("action", (工具名, 原始输入, 解析后的参数 dict 或 None))
    - ("final", 最终答案文本)
    - ("other", 原文本)——没有 Action 也没有 Final Answer 时的兜底。
    """
    m = _ACTION_RE.search(text)
    if m:
        name = m.group("name")
        raw = m.group("input")
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            args = None
        return ("action", (name, raw, args))
    fm = _FINAL_RE.search(text)
    if fm:
        return ("final", fm.group("final").strip())
    return ("other", text)


def run_react(
    question: str,
    max_steps: int = 10,
    stream: bool = False,
    on_delta=None,
) -> str:
    """ReAct 循环:Thought -> Action -> Observation,直到 Final Answer。

    注意:纯文本协议没有 tool_call_id 配对机制,模型可能重复发相同 Action,
    所以这里同样用幂等去重(tools.IDEMPOTENT_TOOLS + tool_key)收敛。
    """
    msgs = [
        {"role": "system", "content": REACT_SYSTEM},
        {"role": "user", "content": question},
    ]
    dedup: dict[str, str] = {}
    last_observation = ""

    for step in range(max_steps):
        text = _call_model(msgs, stream=stream, on_delta=on_delta)
        msgs.append({"role": "assistant", "content": text})
        log_event("react_output", "", step=step, chars=len(text))

        kind, payload = _parse(text)
        if kind == "final":
            log_event("react_final", "", step=step, answer=payload[:200])
            return payload
        if kind == "other":
            # 没有 Action 也没有 Final Answer:把模型输出当作最终回答兜底。
            return text

        name, input_raw, args = payload
        if args is None:
            result = f"Action Input 不是合法 JSON: {input_raw}"
        else:
            key = tool_key(name, args) if name in IDEMPOTENT_TOOLS else None
            if key is not None and key in dedup:
                result = dedup[key]
                log_event(
                    "react_action",
                    f"[react] {name}({input_raw}) -> (幂等去重:复用)",
                    tool=name, dedup=True, step=step,
                )
            else:
                try:
                    result = run_tool(name, args)
                except Exception as exc:
                    result = f"工具执行出错: {exc}"
                if key is not None and not result.startswith("工具执行出错"):
                    dedup[key] = result
                log_event(
                    "react_action",
                    f"[react] {name}({input_raw})",
                    tool=name, arguments=input_raw, step=step,
                )

        last_observation = result
        log_event(
            "react_observation",
            f"[react] Observation: {result[:80]}",
            step=step, result=result[:200],
        )
        # Observation 以用户消息回填,驱动下一轮 Thought。
        msgs.append({"role": "user", "content": f"Observation: {result}"})

    return (
        f"达到最大步数({max_steps}),未收敛。\n"
        f"最后一条 Observation:\n{last_observation}"
    )
