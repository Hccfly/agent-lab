# agent-lab
一个不依赖 LangChain 等任何 Agent 框架、从零实现的 agent 系统。工具循环、会话记忆与摘要压缩、 RAG 检索、多 agent 协作、ReAct、幂等去重、流式、结构化日志、HTTP/SSE 服务化、eval 质量层—— 每一层都摊开自己实现,不把原理封装进框架抽象里。  三种执行范式(原生 tool calling / Planner+Executor / ReAct)、两级长期记忆、多专用 Executor 协作、 失败自愈重派、拓扑并行、eval 端到端打分。**9 个测试文件全不调 API**(mock 模型),`python evaluate.py` 真调 API 做端到端评测。
