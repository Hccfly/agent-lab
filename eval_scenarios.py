"""共用评测场景与确定性模型：不从真实模型输出推导参考答案。"""
import json
import re
from types import SimpleNamespace


CASES = [
    dict(id="dialogue", mode="single", goal="不要调用工具，只回复：你好。",
         answer="你好", expressions=[], live=True),
    dict(id="arithmetic", mode="single", goal="请用 calculator 计算 127*53，只回复结果。",
         answer="6731", expressions=["127*53"], live=True),
    dict(id="dedup", mode="single", goal="同一轮两次调用 calculator 计算 127*53，最后回复结果。",
         answer="6731", expressions=["127*53"], requested=2),
    dict(id="tool_retry", mode="single", goal="用 calculator 计算 127*53，临时失败则重试一次。",
         answer="6731", expressions=["127*53", "127*53"], fault="temporary"),
    dict(id="tool_timeout", mode="single", goal="用 calculator 计算 127*53，超时则重试一次。",
         answer="6731", expressions=["127*53", "127*53"], fault="timeout"),
    dict(id="loop_limit", mode="single", goal="持续查询当前时间。",
         answer="最大迭代次数", expressions=[None] * 3, expected_stop="max_iterations"),
    dict(id="model_error", mode="single", goal="你好", answer="", expressions=[],
         expected_stop="error"),
    dict(id="multi_diamond", mode="multi",
         goal="拆成三个子任务：先分别用 calculator 计算 12*5 和 7*8，最后根据这两个结果用 calculator 计算两者之和。返回所有结果。",
         answer="116", expressions=["12*5", "7*8", "60+56"], live=True),
]


PLAN = {"tasks": [
    {"task_id": 1, "agent": "calculator", "description": "JOB1: 计算 12*5", "depends_on": []},
    {"task_id": 2, "agent": "calculator", "description": "JOB2: 计算 7*8", "depends_on": []},
    {"task_id": 3, "agent": "calculator", "description": "JOB3: 根据前置结果计算 60+56", "depends_on": [1, 2]},
]}


class Message:
    def __init__(self, content="", calls=()):
        self.content = content
        self.tool_calls = [SimpleNamespace(
            id=f"call-{i}", type="function",
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        ) for i, (name, args) in enumerate(calls)]

    def model_dump(self, **_kwargs):
        data = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [dict(id=c.id, type="function", function=dict(
                name=c.function.name, arguments=c.function.arguments)) for c in self.tool_calls]
        return data


class ScenarioClient:
    """按输入消息驱动，同一场景两版使用相同规则，离线 usage 不伪造。"""
    def __init__(self, case):
        self.case = case
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        from multiagent import PLANNER_SYSTEM
        from langgraph_multiagent import DAG_PLANNER_SYSTEM, SUMMARIZER_SYSTEM
        messages = kwargs["messages"]
        system = messages[0]["content"]
        case = self.case
        if system in (PLANNER_SYSTEM, DAG_PLANNER_SYSTEM):
            filled = any("已完成子任务的结果:" in m.get("content", "") for m in messages[1:])
            msg = Message("60，56，总和 116" if filled else json.dumps(PLAN))
        elif system == SUMMARIZER_SYSTEM:
            msg = Message("60，56，总和 116")
        elif case["id"] == "model_error":
            raise RuntimeError("injected model failure")
        elif case["id"] == "dialogue":
            msg = Message("你好")
        elif case["id"] == "loop_limit":
            msg = Message(calls=[("get_current_time", {})])
        else:
            observations = [m for m in messages if m["role"] == "tool"]
            if not observations or (case.get("fault") and len(observations) == 1):
                expression = "127*53"
                if case["mode"] == "multi":
                    tid = int(re.search(r"JOB(\d+)", messages[1]["content"])[1])
                    expression = ["12*5", "7*8", "60+56"][tid - 1]
                calls = [("calculator", {"expression": expression})]
                msg = Message(calls=calls * case.get("requested", 1))
            else:
                msg = Message(observations[-1]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)
