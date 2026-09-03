"""工具层:定义工具 schema,并实现工具执行逻辑。

工具 schema(TOOLS)就是交给模型的"使用说明书"。
设计要点:描述写得越具体,模型越会用对工具、越少幻觉参数。
"""
import ast
import datetime
import json
import operator
from pathlib import Path

def _tool(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    """构造 Chat Completions 格式的工具定义。

    注意:Chat Completions 要求工具包在 {"type": "function", "function": {...}}
    里,而 Responses API 是平铺的 {"type": "function", "name": ..., "parameters": ...}。
    用哪个 API,格式必须匹配,否则服务端报 missing field 'function'。
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        },
    }


TOOLS = [
    _tool(
        "get_current_time",
        "获取当前本地日期和时间,例如 2026-08-31 18:30:00 星期一。"
        "当用户问'现在几点''今天几号''今天星期几'时使用。",
        {},
    ),
    _tool(
        "calculator",
        "执行安全的数学计算,支持 + - * / % 和括号。"
        "当用户给出算术表达式(例如 '3 + 5 * 2')时使用,而不是自己心算。",
        {
            "expression": {
                "type": "string",
                "description": "数学表达式,例如 '(3 + 5) * 2'",
            }
        },
        ["expression"],
    ),
    _tool(
        "read_file",
        "读取本机文本文件的内容(最多前 4000 字符)。"
        "当用户想了解某个文件的内容时使用。",
        {
            "path": {
                "type": "string",
                "description": "文件的绝对路径,例如 D:/实习/agent-lab/README.md",
            }
        },
        ["path"],
    ),
]

# 只允许白名单内的运算符,禁止任何函数/属性访问。
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    """对数学表达式做白名单求值,而不是 eval()。

    设计要点:eval() 能执行任意代码。模型可能被 prompt 诱导输出
    'eval(\"__import__('os').system('rm -rf /')\")' 这类恶意参数,
    白名单求值从根上杜绝了这种风险。
    """
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("表达式包含不支持的语法")


def calculator(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    return str(_safe_eval(tree))


def get_current_time() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


MAX_FILE_CHARS = 4000


def read_file(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        return f"文件不存在或不是文件: {path}"
    # 限制读取长度,防止工具返回超长内容刷爆模型上下文(token 成本问题)。
    return p.read_text(encoding="utf-8", errors="replace")[:MAX_FILE_CHARS]


def run_tool(name: str, arguments: dict) -> str:
    """工具分发器:把模型的调用意图路由到具体实现。"""
    if name == "get_current_time":
        return get_current_time()
    if name == "calculator":
        return calculator(arguments["expression"])
    if name == "read_file":
        return read_file(arguments["path"])
    return f"未知工具: {name}"


# 幂等工具:同参数重复调用结果相同、无副作用,允许在 agent 循环内做结果去重。
# 非幂等工具(如 get_current_time,每次结果不同)每次必须真执行。
IDEMPOTENT_TOOLS = frozenset({"calculator", "read_file"})


def tool_subset(*names: str) -> list[dict]:
    """从 TOOLS 按工具名抽子集,用于专用 executor 的能力边界。

    多专用 agent 时,每个角色只拿到自己该有的工具:
    - 能力最小化(最小权限):文件 agent 拿不到 calculator,天然不能越权计算;
    - 上下文聚焦:模型看到的工具少了,就不会幻想着调用能力之外的"工具"。
    """
    return [t for t in TOOLS if t["function"]["name"] in names]


def tool_key(name: str, arguments: dict) -> str:
    """计算一次工具调用的规范化指纹,用于幂等去重。

    参数按 key 排序后序列化,使内容相同但键顺序不同的参数
    ({"path": "a"} vs {"path": "a"}) 被视为同一调用。
    设计要点:指纹算法是"同参数"的精确判据,排序保证参数顺序无关。
    """
    return f"{name}|{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"
