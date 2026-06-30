# DeepSeek Agent Project

基于 DeepSeek API（OpenAI 兼容协议）构建的个人 AI 助手。支持工具调用、多轮对话持久化、子代理并行执行、语义缓存、MCP 服务器集成（14 个社区工具）和 FastAPI + SSE Web 界面。

## 项目结构

```
Agent_Project/
├── agent.py              # Agent 主体（终端版）：对话循环 + 工具调用 + 记忆恢复
├── agentv2.py            # Agent 主体（流式版）：run_stream() 生成器，供 server.py 调用，含 MCP 集成
├── tools.py              # 基础设施：ToolRegistry + 7 个工具函数 + SessionStore + SubagentPool + SemanticCache
├── mcp_client.py         # MCP 客户端：同步 subprocess 实现，自行拼 JSON-RPC 报文
├── mcp_manager.py        # MCP 管理器：多服务器发现、工具前缀路由、同步调用
├── server.py             # FastAPI 服务：SSE 流式端点、多会话管理
├── index.html            # Web 聊天界面：深色主题、Markdown 渲染、多会话切换
├── chat.py               # 纯对话 demo（无工具能力，阶段 0 产物）
├── test_all_features.py  # 集成测试脚本（19 项覆盖全部功能）
├── requirements.txt      # 精确版本依赖
├── .env                  # API 密钥（不提交）
├── .gitignore
└── README.md
```

## 已实现的能力

### 核心循环（agent.py / agentv2.py）

- **ReAct 循环**：模型输出→如有工具调用则执行→结果回喂→继续推理，最多 15 轮
- **流式输出**：`run_stream()` 生成器 + SSE 推送到浏览器
- **会话持久化**：每轮对话写入 `agent_sessions.json`，支持跨轮次记忆恢复
- **语义缓存**：命中相似问题（余弦相似度 ≥ 0.85）时直接返回缓存回答，跳过 API 调用
- **MCP 集成**：自动发现并注册社区 MCP 服务器工具，支持多服务器共存与工具名前缀路由
- **子代理**：通过 `subagent_task` 工具暴露给模型，支持独立上下文执行 + ThreadPoolExecutor 并行

### 工具系统

#### 本地工具（tools.py）

| 工具名 | 功能 |
|--------|------|
| `calculator` | 安全数学表达式计算 |
| `get_weather_tool` | 模拟天气查询（写死数据，仅测试用） |
| `read_file_tool` | 读取文件内容 |
| `write_file_tool` | 写入文件，自动创建父目录 |
| `shell_tool` | 执行 shell 命令（含危险命令黑名单拦截） |
| `search_files_tool` | 按通配符搜索文件名 |
| `grep_tool` | 在 .py/.ts/.md 文件中搜索文本 |
| `subagent_task` | 启动子代理执行独立任务 |

#### MCP 工具（mcp_client.py / mcp_manager.py）

通过 `subprocess.Popen` 启动 MCP 服务器子进程，自行拼 JSON-RPC 报文，零 async/anyio 依赖。

| 前缀 | 来源 | 注册工具数 |
|------|------|:---------:|
| `filesystem_*` | `@modelcontextprotocol/server-filesystem` | 14 |

### Web 界面（server.py + index.html）

- **FastAPI 服务**：SSE 流式推送到浏览器，缓冲区机制（标点断句/50 字兜底）
- **深色主题界面**：Markdown 渲染、多会话切换
- 访问 `http://localhost:8000`

## 运行依赖

```bash
# 核心依赖（所有模式都需要）
pip install openai==2.44.0 python-dotenv==1.2.2
# Web 界面模式额外需要
pip install fastapi uvicorn sse-starlette
```

## 运行方式

### 终端模式（带 MCP，推荐）

```bash
pip install openai==2.44.0 python-dotenv==1.2.2
python agentv2.py
```

启动后自动连接 MCP 服务器并注册社区工具。

### 终端模式（纯本地工具，无 MCP）

```bash
python agent.py
```

### Web 界面模式

```bash
pip install openai==2.44.0 python-dotenv==1.2.2 fastapi uvicorn sse-starlette
python server.py
```

浏览器访问 `http://localhost:8000`。支持多会话切换、流式输出、Markdown 渲染。

会话 ID 固定为 `"1"`，重启后自动恢复上次对话历史。

## 已知待改进项

- 语义缓存的 `_embed()` 使用空格分词，对中文无效（需要 `jieba` 或字符级方案）
- 无成本追踪（未记录每次 API 调用的 token 用量）
- 主 Agent 和子代理共享同一 API key 和模型配置

---

## 版本记录

### v2.0 — MCP 集成（当前版本）

**新增文件：**
- `mcp_client.py` — 纯 `subprocess.Popen` 实现的 MCP 客户端，零 async/anyio 依赖
- `mcp_manager.py` — 多服务器管理器，自动发现工具、前缀路由

**能力提升：**
- 工具数从 8 个扩展到 8 + N 个，N 等于社区 MCP 服务器数量
- 已接入 `@modelcontextprotocol/server-filesystem`，注册 14 个文件系统工具
- 工具名前缀防止多服务器冲突：`filesystem_read_file`、`github_create_issue`

**解决的关键问题：**
- 放弃 MCP SDK（`anyio` 在 Windows 上 cancel scope 崩溃）
- 放弃 `asyncio.create_subprocess_exec`（Windows 不搜索 PATH）
- 放弃 `asyncio.run()` 桥接（多事件循环导致 disconnect 时 hang）
- 最终方案：同步 `subprocess.Popen` + 手动拼 JSON-RPC
- `__del__` 中不做任何 IO（Python 退出时事件循环不可用）

---

### v1.3 — Web 界面（FastAPI + SSE）

**新增文件：** `server.py`、`index.html`

**能力提升：**
- 从终端升级为 HTTP 服务，浏览器访问 `http://localhost:8000`
- SSE 流式推流 + 缓冲区机制（标点断句 / 50 字兜底）
- 多会话切换，每个会话独立保存历史

**文件变更：**
- 新增 `run_stream()` 生成器方法（agentv2.py）
- 新增依赖 `fastapi uvicorn sse-starlette`

---

### v1.2 — 语义缓存 + 子代理

**新增能力：**
- `SemanticCache`：词袋嵌入 + 余弦相似度，命中时跳过 API
- `SubagentPool`：独立上下文子代理 + `ThreadPoolExecutor` 并行

**修复：**
- assistant(tool_calls) 存入 SessionStore，恢复时不再报错
- JSON 文件用 `encoding="utf-8"`，解决 GBK 编码崩溃

---

### v1.1 — 会话持久化

- `SessionStore`：JSON 文件持久化，跨轮次记忆恢复
- `run()` 接收 `session_id` 参数

---

### v1.0 — 初版

- 基础 ReAct 循环 + 8 个手写工具 + `ToolRegistry` + 终端交互
