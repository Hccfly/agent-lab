# Human-in-the-loop 敏感工具审批

## 1. 目标与结果

当前实现为 `read_file` 增加了真正位于工具执行前的人工审批点。模型可以提出调用，但系统会先把
`tool name + arguments + call_id` 写入 SQLite 并暂停；只有收到同一 `thread_id` 的审批命令后，
才会执行、修改后执行，或拒绝执行。

三种决策语义：

- `approve`：按模型原参数执行；
- `edit`：丢弃模型原参数，只执行人工提交的新参数；
- `reject`：完全不调用工具，把拒绝原因作为 Tool Observation 回填模型。

`calculator`、`get_current_time` 等未列入敏感策略的工具保持自动执行。

## 2. 执行流程

```text
用户问题
  -> model 产生 read_file tool_call
  -> tools 节点识别 sensitive_tools
  -> interrupt({calls, allowed_actions})
  -> SqliteSaver 保存状态并返回 CLI
  -> 人工查看 name/arguments
       ├─ approve ─> 原参数执行工具
       ├─ edit ────> 新参数执行工具
       └─ reject ──> 不执行，拒绝原因回填模型
  -> model 根据 Tool Observation 生成最终回答
```

## 3. 为什么 interrupt 必须放在真实调用之前

LangGraph 恢复动态 interrupt 时，会从暂停节点的开头重新执行，而不是从 Python 函数的暂停行继续。
因此 `tool_node` 的顺序严格是：

```python
response = interrupt(review_payload)
decisions = validate(response)

# 所有真实调用都在 interrupt 之后
result = tool_runner(name, approved_arguments)
```

节点重进时，`interrupt()` 会返回 `Command(resume=...)` 提供的值。由于它之前只有状态筛选和
JSON payload 构造，没有文件、网络或数据库业务写操作，所以不会因为重进产生重复副作用。

不能用普通 `input()` 代替：`input()` 会占住当前进程，无法跨服务实例恢复，也不会把待审批内容
纳入图状态。动态 interrupt 则与 checkpoint、`thread_id` 和执行轨迹属于同一个状态机。

## 4. 状态与依赖边界

`AgentDependencies.sensitive_tools` 是可注入策略，默认单 Agent 图保持原有无状态行为；
`PersistentApprovalAgent` 为 HITL 入口注入 `{"read_file"}`、SQLite checkpointer 和稳定
`thread_id`。这样离线测试可以使用 FakeClient/FakeToolRunner，不访问真实模型或文件。

一次 interrupt 中的所有敏感调用必须原子提交完整决策：不能漏掉、重复或审批未知 `call_id`。
`edit` 参数必须是 JSON 对象，`reject` 必须携带非空原因。每个工具轨迹增加 `approval` 字段：
`not_required / approve / edit / reject`。

## 5. CLI 演示

启动：

```powershell
.venv\Scripts\python.exe main.py
```

请求读取文件：

```text
/hitl-start file-review-1 | 请读取 D:/实习/agent-lab/README.md 并总结
```

查看待审批调用：

```text
/hitl-status file-review-1
```

任选一种决策：

```text
/hitl-approve file-review-1
/hitl-reject file-review-1 | 该文件不在本次任务授权范围
/hitl-edit file-review-1 | {"path":"D:/实习/agent-lab/docs/SQLite断点恢复.md"}
```

每个示例应使用新的 `thread_id`。数据库默认位于：

```text
checkpoints/hitl_state.sqlite3
```

## 6. 离线验收

```powershell
.venv\Scripts\python.exe -X utf8 tests\test_hitl.py
```

测试覆盖：

1. 进程生命周期切换前，待审批 `read_file` 的真实调用次数为 0；
2. approve 恢复后只执行一次原参数；
3. reject 不执行工具，并把原因回填模型；
4. edit 只执行修改后的参数；
5. 非敏感 calculator 不触发 interrupt。

## 7. 设计总结

> 我没有在 CLI 外层用 input 阻塞，而是把审批建模为 LangGraph 动态 interrupt。模型产生
> `read_file` 调用后，工具节点先把可序列化的调用信息交给 interrupt，由 SQLite checkpoint
> 保存执行位置；审批端用相同 thread_id 和 `Command(resume=...)` 提交 approve、edit 或 reject。
> 因为 LangGraph 恢复时会重跑整个节点，我把所有真实副作用放到 interrupt 之后。拒绝也不是直接
> 丢弃请求，而是构造成 Tool Observation 回填模型，让模型能向用户解释或选择替代方案。

当前提供独立 `/hitl-*` 单 Agent 审批入口；原有 `/graph`、`/graph-multi` 用于对照，
不会自动启用该策略。生产部署仍应把策略下沉为所有入口不可绕过的统一授权层。
