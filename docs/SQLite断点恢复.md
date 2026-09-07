# SQLite checkpoint 与跨进程断点恢复

## 1. 今天解决的问题

并行图即使能正确执行，只要 Python 进程退出，内存中的 `task_status`、`task_results`
和下一待执行节点就会消失。当前实现将父图的每个 superstep 写入 SQLite，并用 `thread_id`
把同一个任务的多次进程启动串成一条执行时间线。

本次交付满足三个验收点：

1. 关闭连接并重新创建模型客户端和图后，仍能读取上一进程的执行状态；
2. 恢复时以 `graph.invoke(None, config)` 从 checkpoint 的 `next` 节点继续，而不是重新传初始输入；
3. 已写入 checkpoint 的 Executor 结果不再执行，已完成线程再次 resume 也是无操作。

## 2. 核心结构

```text
start(goal, thread_id)
    │ 完整 initial_state
    ▼
LangGraph 父图 ── 每个 superstep ──> SqliteSaver ──> checkpoints/agent_state.sqlite3
    │                                      ▲
    │ executor 波次后静态暂停              │ 相同 thread_id
    ▼                                      │
进程退出                         新进程 resume(thread_id)
                                           │ invoke(None, config)
                                           ▼
                                 checkpoint.next -> 后续节点
```

`checkpointing.py` 持有 SQLite 连接、`SqliteSaver` 和编译后的父图；
`langgraph_multiagent.build_multiagent_graph()` 只接收注入的 checkpointer，不直接决定存储实现。
因此普通 `/graph-multi` 保持无状态运行，checkpoint Demo 才启用 SQLite。

## 3. 从内存图到持久图的代码步骤

### 步骤一：编译图时注入 checkpointer

```python
checkpointer = SqliteSaver(sqlite3.connect(path, check_same_thread=False))
graph = build_multiagent_graph(
    dependencies,
    checkpointer=checkpointer,
    interrupt_after=["executor"],
)
```

SQLite 连接使用 `check_same_thread=False`，因为同一波的多个 `Send("executor", ...)`
可能在线程池中并发完成；`SqliteSaver` 内部负责串行化数据库访问。

### 步骤二：所有调用携带稳定 thread_id

```python
config = {
    "configurable": {"thread_id": "checkpoint-demo"},
    "recursion_limit": 20,
    "max_concurrency": 4,
}
```

`thread_id` 不是展示字段，而是 checkpoint 的主检索键。换一个 ID 就是一条完全隔离的新状态线。
入口只接受字母、数字、点、下划线和短横线，长度不超过 128。

### 步骤三：首次执行传状态，恢复执行传 None

```python
# 首次启动
graph.invoke(initial_multiagent_state(goal, max_tasks), config)

# 同一线程恢复
graph.invoke(None, config)
```

恢复时如果再次传 `initial_state`，语义是给现有线程追加一轮输入，不能表示“从断点继续”。
因此 `PersistentMultiAgent.start()` 会拒绝已有 ID，`resume()` 只允许已有 ID。

### 步骤四：用 StateSnapshot 判断暂停还是完成

```python
snapshot = graph.get_state(config)
completed = not snapshot.next
```

`snapshot.values` 是最近状态，`snapshot.next` 是下一批待运行节点。静态断点配置为
`interrupt_after=["executor"]` 后，每完成一个 Executor 波次便返回 CLI；再次 resume 后从 scheduler 开始，
不会重新进入刚完成的 Executor。

### 步骤五：完成态 resume 保持幂等

`PersistentMultiAgent.resume()` 先读快照。如果 `snapshot.next` 为空，直接返回已保存答案，
不调用 `graph.invoke()`。这是入口层的防误操作，与工具循环内部的参数指纹去重是两层不同保障。

## 4. 可运行演示

```powershell
.venv\Scripts\python.exe main.py
```

在 CLI 中输入：

```text
/checkpoint-start checkpoint-demo | 计算 12*5，并根据结果完成后续说明
/checkpoint-status checkpoint-demo
/checkpoint-resume checkpoint-demo
```

如果 Planner 生成多个依赖波次，每次 `/checkpoint-resume checkpoint-demo` 推进到下一个
Executor 波次断点；输出出现“已完成”时，最终答案已经写入 SQLite。此时退出 `main.py`、
重新启动后再次 status/resume，状态仍然存在。默认数据库为：

```text
checkpoints/agent_state.sqlite3
```

运行自动验收：

```powershell
.venv\Scripts\python.exe -X utf8 tests\test_checkpoint_resume.py
```

测试全程屏蔽 socket：第一次进程对象执行 task 1/2 后关闭；第二个对象只执行 task 3；
第三个对象只运行汇总；最后再次 resume，工具总调用数仍为 3。

## 5. 恢复语义边界

checkpoint 提供的是“从最近成功 superstep 恢复”，并不自动把任意外部副作用变成严格 exactly-once。
如果进程在某个工具已经写入外部系统、但当前节点结果尚未提交 checkpoint 的瞬间崩溃，该节点恢复后
仍可能重试。生产系统应给写操作传业务幂等键，并由数据库唯一约束、事务或外部服务幂等接口兜底。

本项目当前能保证的是：

- checkpoint 已确认完成的 Executor 波次不会重跑；
- 同一个已存在 ID 不能误用 `start()` 覆盖；
- 完成态 `resume()` 不产生模型或工具调用；
- 单次 Agent 内的幂等工具仍使用参数指纹缓存。

这个边界比笼统宣称“用了 checkpoint 就绝不重复副作用”更准确，也是继续做安全与并发治理的基础。
