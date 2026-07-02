# DeepSeek Agent Project

基于 DeepSeek API（OpenAI 兼容协议）构建的个人 AI 助手。支持工具调用、多轮对话持久化、子代理并行执行、语义缓存、**RAG 知识库检索**、MCP 服务器集成（38 个社区工具）和 FastAPI + SSE Web 界面。

## 项目结构

```
NN_text1/
├── agent.py              # Agent 主体（纯本地工具）
├── agentv2.py            # Agent 主体（MCP + 工具注册 + 流式生成器）
├── tools.py              # 基础设施：ToolRegistry + 9 个工具函数 + SessionStore + SubagentPool + SemanticCache
├── rag_tool.py           # RAG 检索工具（懒加载版，Chromadb + sentence-transformers）
├── mcp_client.py         # MCP 客户端：同步 subprocess，手动 JSON-RPC
├── mcp_manager.py        # MCP 管理器：多服务器发现、前缀路由
├── server.py             # FastAPI 服务：SSE 流式端点、多会话管理
├── index.html            # Web 聊天页面：深色主题、多会话切换
├── requirements.txt      # Python 依赖
├── .env                  # API 密钥（需自行创建）
└── README.md
```

## 已实现的能力

### 核心循环（agent.py / agentv2.py）

- **ReAct 循环**：模型输出 → 工具调用 → 结果回喂 → 继续推理，最多 15 轮
- **流式输出**：`run_stream()` 生成器 + SSE 推送到浏览器（逐字显示）
- **会话持久化**：每轮对话写入 `agent_sessions.json`，跨轮次记忆恢复
- **语义缓存**：命中相似问题（余弦相似度 ≥ 0.85）时直接返回缓存回答
- **MCP 集成**：自动发现并注册社区 MCP 服务器工具，多服务器共存
- **子代理**：通过 `subagent_task` 暴露给模型，支持独立上下文 + 并行执行
- **RAG 知识库**：注册为工具，检索后交由主 LLM 自行判断是否采用

### 工具系统

#### 本地工具（tools.py）

| 工具名 | 功能 |
|--------|------|
| `calculator` | 安全数学表达式计算 |
| `get_weather_tool` | 模拟天气查询 |
| `read_file_tool` | 读取文件内容 |
| `write_file_tool` | 写入文件，自动创建父目录 |
| `shell_tool` | 执行 shell 命令（含危险命令黑名单） |
| `search_files_tool` | 按通配符搜索文件名 |
| `grep_tool` | 在 .py/.ts/.md 中搜索文本 |
| `subagent_task` | 启动子代理执行独立任务 |
| `rag_query` | 从知识库检索相关段落（sentence-transformers + ChromaDB） |

#### MCP 工具（mcp_client.py / mcp_manager.py）

同步 `subprocess.Popen` 实现，自行拼 JSON-RPC 报文，零 async/anyio 依赖。

| 前缀 | 来源 | 注册工具数 |
|------|------|:---------:|
| `filesystem_*` | `@modelcontextprotocol/server-filesystem` | 14 |
| `notion_API-*` | `@notionhq/notion-mcp-server` | 24 |
| **合计** | | **38** |

### RAG 知识库

- **嵌入模型**：`BAAI/bge-small-zh-v1.5`（中文优化，512 维）
- **向量库**：ChromaDB 持久化存储
- **阈值筛选**：距离 > 0.7 自动丢弃，节省 Token
- **懒加载**：首次调用 `rag_query` 时加载模型，启动不阻塞
- **配合 System Prompt**：主 LLM 自行判断检索内容是否相关

### Web 界面（server.py + index.html）

- **FastAPI 服务**：SSE 流式推送，缓冲区机制（标点断句 / 50 字兜底）
- **深色主题**：Markdown 渲染、多会话切换、对话历史保留
- 访问 `http://localhost:8000`

## 运行依赖

```bash
# 核心依赖
pip install openai==2.44.0 python-dotenv==1.2.2

# RAG 知识库
pip install sentence-transformers chromadb

# Web 界面
pip install fastapi uvicorn sse-starlette

# MCP 服务器（按需）
npx @modelcontextprotocol/server-filesystem    # filesystem 工具
npx @notionhq/notion-mcp-server                # Notion 集成（需 API Key）
```

## 运行方式

### 终端模式（推荐）

```bash
pip install openai==2.44.0 python-dotenv==1.2.2
python agentv2.py
```

自动连接 MCP 服务器 + 注册本地工具。

### Web 界面模式

```bash
pip install openai==2.44.0 python-dotenv==1.2.2 fastapi uvicorn sse-starlette
python server.py
```

浏览器访问 `http://localhost:8000`。支持多会话切换、流式输出、Markdown 渲染。

### RAG 环境验证

```bash
# 国内用户需设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 验证 RAG 组件
pip install sentence-transformers chromadb
python RAG_learning/rag_demo.py
```

## 环境变量

创建 `.env` 文件：

```
DEEPSEEK_API_KEY=sk-your-key-here
```

## 已知待改进项

- 语义缓存的 `_embed()` 使用空格分词，对中文效果差
- 无成本追踪（未记录每次 API 调用的 token 用量）
- RAG 知识库目前为硬编码文档，未接入外部文件加载
- MCP 的 `readline()` 无超时机制，某些场景可能阻塞

---

## 版本记录

### v2.1 — RAG 知识库集成（当前版本）

**新增文件：**
- `rag_tool.py` — RAG 检索工具（懒加载、阈值筛选、返回原文）
- `RAG_learning/` — 4 个入门练习文件
- `学习计划-自己动手写RAG.md` — 8 个动手练习，从嵌入到 Rerank
- `RAG学习路线清单.md` — RAG 学习路线总览

**能力提升：**
- Agent 新增 `rag_query` 工具，支持知识库语义检索
- 模型延迟加载：启动时不阻塞，首次调用工具时才加载
- 返回原文而非 LLM 生成结果，主 LLM 自行判断相关度
- 距离阈值（≤ 0.7 保留）自动过滤不相关内容

**依赖新增：**
- `sentence-transformers` — 本地嵌入模型
- `chromadb` — 向量数据库

**注意：**
- RAG的集成表示需要安装 pytorch 等神经网络训练相关的库
- 最好能够使用带有 NVIDIA GPU 的 PC 主机进行尝试
- AMD GPU 的 PC 主机可以尝试使用 WSL2 ，启用 Linux 环境进行运行 [配置教程](https://github.com/violetwhale-or/Using-AMD-GPU-with-PyTorch-on-Windows-and-WSL2)
- 也可以安装 CPU 版本的 pytorch 进行尝试

---

### v2.0 — MCP 集成

**新增文件：** `mcp_client.py`、`mcp_manager.py`

**能力提升：**
- 工具数从 8 个扩展到 8 + N 个
- 已接入 filesystem（14 工具）+ notion（24 工具），共 38 个 MCP 工具
- 工具名前缀防止多服务器冲突

**解决的关键问题：**
- 放弃 asyncio 方案（Windows 兼容性），最终采用同步 subprocess + 手动 JSON-RPC

---

### v1.3 — Web 界面

**新增文件：** `server.py`、`index.html`

**能力提升：**
- 终端 → HTTP 服务，浏览器访问 `http://localhost:8000`
- SSE 流式推送 + 缓冲区机制

---

### v1.2 — 语义缓存 + 子代理

**新增：** `SemanticCache`、`SubagentPool`

---

### v1.1 — 会话持久化

**新增：** `SessionStore`、`session_id` 参数

---

### v1.0 — 初版

基础 ReAct 循环 + 8 个手写工具 + `ToolRegistry`
