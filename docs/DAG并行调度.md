# 把任务依赖映射为 LangGraph 并行分支与合流

## 从串行 Executor 到依赖并行

早期实现用 `next_task_index` 依次取任务；当前实现根据 `depends_on` 找出所有就绪任务，
通过 LangGraph `Send` 并行调用 Executor 子图，整波执行结束后再计算下一波。
核心实现位于 `langgraph_multiagent.py`，没有另外建立 Python ThreadPoolExecutor。

例如 Planner 输出任务 3 在前面也不影响调度：

```json
{"tasks": [
  {"task_id": 3, "agent": "calculator", "description": "将两份结果相加", "depends_on": [1, 2]},
  {"task_id": 1, "agent": "calculator", "description": "计算 12*5", "depends_on": []},
  {"task_id": 2, "agent": "calculator", "description": "计算 7*8", "depends_on": []}
]}
```

第一波任务 1、2 并发；合流后第二波任务 3 收到两份真实结果。最后只调用一次 Summarizer。

## 源码阅读顺序

1. `_normalize_tasks`：校验任务 ID 唯一、依赖存在、无环。超出 `max_tasks` 时拒绝整个计划，
   防止截断把依赖节点丢掉。校验失败返回 `invalid_plan`，不会执行任何工具。
2. `MultiAgentState`：`ready_tasks` 存就绪集，`wave` 存波次，移除串行游标。
   `task_results` 仅存成功结果；`task_errors` 存失败和受阻原因；`task_status` 存终态。
3. `scheduler_node`：只把所有直接依赖均为 `completed` 的任务列为就绪。
   失败后继通过循环传播 `blocked`，无关分支保持可执行。
4. `dispatch_ready`：返回 `Send("executor", {task, prior_results, wave})` 列表。
   不同分支拿到不同任务和独立输入字典；只注入显式声明的直接前置结果。
5. `executor_node`：调用 Executor 子图，保留基础模型/工具闭环与每任务缓存。
   各分支只返回 `{本任务ID: 值}`，不能复制整个父结果表回写。
6. `merge_task_maps`：使用新字典合并增量结果，不原地改共享状态；冲突 ID 抛错。
7. `executor -> scheduler`：所有 Send 分支在一个父图 superstep 中执行，整波完成后
   Reducer 合并输出，再唤醒 scheduler。这是本项目的合流屏障。
8. `summarizer_node`：没有就绪任务且所有任务已有终态后，汇总完整状态表。

实现参考：[LangGraph Send API](https://reference.langchain.com/python/langgraph/types/Send)
与 [Graph API 的 Map-Reduce 示例](https://docs.langchain.com/oss/python/langgraph/use-graph-api)。

## 为什么需要 Reducer

原来的 `task_results: dict` 默认是覆盖语义，同一步多个分支写入会冲突。
改为 `Annotated[dict[int, str], merge_task_maps]` 后：

```python
# 第一波两个分支输出
{"task_results": {1: "60"}}
{"task_results": {2: "56"}}
# 合流后父状态
{"task_results": {1: "60", 2: "56"}}
```

`trace` 仍使用列表追加 Reducer；合并顺序不是实际执行时间顺序。通过 `wave + task_id`
识别同波分支，不能把拼接后的列表当作串行调用栈。

## 失败行为

`call_role` 将模型调用异常、最大模型轮次、空答案及既有文本失败信号转为子任务失败。
保留一次 general 重派；仍失败则写入 `task_errors`，不污染成功结果表。

例如 1 失败、2 独立、3 依赖 1、4 依赖 3：

```python
task_status = {1: "failed", 2: "completed", 3: "blocked", 4: "blocked"}
stop_reason = "partial_failure"
```

3、4 不执行，2 的成功结果保留。CLI 逐任务打印错误，Summarizer 也收到状态和原因。

## 运行与验收

在项目根目录运行：

```powershell
.venv\Scripts\python.exe -X utf8 tests\test_langgraph_dag.py
.venv\Scripts\python.exe main.py
```

进入 CLI 后输入（该命令使用真实模型 API）：

```text
/graph-multi 分别计算12*5和7*8，然后基于这两个结果求和。请声明求和任务依赖前两个任务。
```

Python 调用时可设置并发上限：

```python
from langgraph_multiagent import invoke_multiagent
state = invoke_multiagent("你的目标", max_tasks=5, max_concurrency=2)
print(state["task_status"], state["task_errors"])
```

`test_langgraph_dag.py` 禁止 socket 连接，注入 FakeClient/Fake Tool Runner。
并发测试让两个任务在 `threading.Barrier(2)` 等待彼此；如果实际串行，屏障超时导致测试失败。
其他断言检查第二波只在第一波完成后启动、多个结果均保留、缓存不跨任务复用、汇总只调用一次。

## 当前边界和设计总结

- 当前一次完整规划，不执行跨批重新规划；Planner 提示已同步为单批 DAG 协议。
- 当前为整波屏障：快分支的后继也需等本波慢分支完成，尚未实现事件驱动提前放行。
- `max_concurrency` 限制单次父图的并发 Executor；注入的模型客户端与工具须支持线程安全调用。
- 文本失败判定仍是启发式，不能证明答案正确；行为质量由独立 Eval 验证。
- 现有 general 重派会扩大工具集；工具 Schema 本身不是执行层权限校验，权限治理留在后续阶段。
- 本轮未实现 checkpoint、人工审批、全局限流、业务超时或新的 HTTP 接口。

设计核心是将 Planner 生成的 DAG 校验后转为动态 Send 分支，按就绪集分波运行。
各分支通过任务 ID Reducer 合并独立结果，再由 scheduler 放行下一波。失败状态与成功结果分离，
失败只阻塞依赖链，独立任务继续完成。测试使用线程屏障验证真实并发，并检查合流、状态隔离和失败传播。
