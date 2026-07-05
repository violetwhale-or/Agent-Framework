# DeepSeek Agent Project

基于 DeepSeek API（OpenAI 兼容协议）构建的个人 AI 助手。支持工具调用、多轮对话持久化、子代理并行执行、语义缓存、**RAG 知识库（混合检索）**、MCP 服务器集成（38 个社区工具）和 FastAPI + SSE Web 界面。

## 项目结构

```
NN_text1/
├── agentv2.py             # Agent 主体：MCP + 工具注册 + 流式生成器
├── agent.py               # Agent 主体（纯本地工具）
├── chat.py                # Agent 对话demo（纯本地工具）
├── tools.py               # ToolRegistry + 9 个工具函数 + SessionStore + SubagentPool + SemanticCache
├── rag_tool.py            # RAG 检索：混合检索（语义+BM25）+ RRF 合并
├── build_knowledge.py     # 知识库构建：md 读取 → 递归切片 → 编码 → 向量库 + BM25 索引
├── mcp_client.py          # MCP 客户端：同步 subprocess，手动 JSON-RPC
├── mcp_manager.py         # MCP 管理器：多服务器发现、前缀路由
├── server.py              # FastAPI 服务：SSE 流式端点、多会话管理
├── index.html             # Web 聊天页面：深色主题、多会话切换
├── Dockerfile             # 容器化镜像
├── docker-compose.yml     # 一键部署
├── requirements.txt       # Python 依赖
├── .env                   # API 密钥（需自行创建）
└── README.md
```

## 已实现的能力

### 核心循环（agentv2.py）

- **ReAct 循环**：模型输出 → 工具调用 → 结果回喂 → 继续推理，最多 15 轮
- **流式输出**：`run_stream()` 生成器 + SSE 推送到浏览器（逐字显示）
- **会话持久化**：每轮对话写入 `agent_sessions.json`，跨轮次记忆恢复
- **语义缓存**：命中相似问题（余弦相似度 ≥ 0.85）时直接返回缓存回答
- **MCP 集成**：自动发现并注册社区 MCP 服务器工具，多服务器共存
- **子代理**：通过 `subagent_task` 暴露给模型，支持独立上下文 + 并行执行
- **RAG 混合检索**：向量语义检索 + BM25 关键词检索 + RRF 合并

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
| `rag_query` | 混合检索知识库（向量 + BM25 + RRF） |

#### MCP 工具（mcp_client.py / mcp_manager.py）

同步 subprocess 实现，自行拼 JSON-RPC 报文，零 async/anyio 依赖。

| 前缀 | 来源 | 注册工具数 |
|------|------|:---------:|
| `filesystem_*` | `@modelcontextprotocol/server-filesystem` | 14 |
| `notion_API-*` | `@notionhq/notion-mcp-server` | 24 |
| **合计** | | **38** |

### RAG 知识库

- **构建与检索分离**：`build_knowledge.py` 独立构建索引，`rag_tool.py` 只做检索
- **递归切片**：按 `##` → `###` → `####` → 段落 层级切割，单块上限 500 字
- **混合检索**：bge-small 语义检索 + BM25 关键词检索 + RRF 排名合并
- **阈值筛选**：距离 > 0.7 自动丢弃，节省 Token
- **懒加载**：首次调用 `rag_query` 时加载模型，启动不阻塞
- **来源标记**：metadata 记录文档来源，检索结果附带文件名

### Web 界面（server.py + index.html）

- **FastAPI 服务**：SSE 流式推送，缓冲区机制（标点断句 / 50 字兜底）
- **深色主题**：Markdown 渲染、多会话切换、对话历史保留
- 访问 `http://localhost:8000`

## 运行依赖

```bash
pip install openai==2.44.0 python-dotenv==1.2.2 fastapi uvicorn[standard] sse-starlette sentence-transformers chromadb rank-bm25
```

或直接：

```bash
pip install -r requirements.txt
```

## 运行方式

### 1. 构建知识库（首次运行前执行）

```bash
# 国内用户需设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 从 .md 文档构建知识库
python build_knowledge.py RAG_learning/Qwen-Proxy.md
```

产物存储在 `./rag_data/`（chroma.sqlite3 + bm25_index.pkl）。后续再次启动 agent 时自动加载。

### 2. 终端模式

```bash
python agentv2.py
```

自动连接 MCP 服务器 + 注册本地工具 + 加载知识库。

### 3. Web 界面模式

```bash
python server.py
```

浏览器访问 `http://localhost:8000`。支持多会话切换、流式输出、Markdown 渲染。

### 4. Docker 部署

```bash
docker compose build
docker compose up -d
```

访问 `http://localhost:8000`。

### 5. 检索效果评估

```bash
python eval_rag.py
```

输出 Recall@K 指标。

## 环境变量

创建 `.env` 文件：

```
DEEPSEEK_API_KEY=sk-your-key-here
```

## 已知待改进项

- 语义缓存的 `_embed()` 使用空格分词，对中文效果差
- 无成本追踪（未记录每次 API 调用的 token 用量）
- Rerank 尚未默认启用（需要数据量 > 1000 条或手动开启）
- MCP 的 `readline()` 无超时机制

---

## 版本记录

### v2.2 — 混合检索 + 部署（当前版本）

**新增文件：**
- `build_knowledge.py` — 独立知识库构建工具（递归切片 + 编码 + 向量库 + BM25 索引）
- `Dockerfile` / `docker-compose.yml` — 容器化部署
- `eval_rag.py` — 检索效果评估脚本

**能力提升：**
- 检索从纯语义升级为混合检索：bge-small 向量 + BM25 关键词 + RRF 合并
- 构建与检索彻底分离，多次构建可增量追加
- Docker 一键部署，不再依赖宿主机 Python 环境
- 评估脚本可量化 Recall@K

**依赖新增：**
- `rank-bm25` — BM25 倒排索引检索

**注意事项：**
- 此版本涉及到Docker部署所以需要Linux （[配置教程](https://github.com/violetwhale-or/Using-AMD-GPU-with-PyTorch-on-Windows-and-WSL2)）环境进行尝试，除此之外windows环境支持运行

---

### v2.1 — RAG 知识库集成

**新增文件：** `rag_tool.py`

**能力提升：**
- Agent 新增 `rag_query` 工具，支持知识库语义检索
- 模型延迟加载，启动不阻塞，首次调用工具时才加载
- 返回原文而非 LLM 生成结果，主 LLM 自行判断相关度
- 距离阈值（≤ 0.7 保留）自动过滤不相关内容

**依赖新增：**
- sentence-transformers — 本地嵌入模型
- chromadb — 向量数据库

**注意事项：**
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