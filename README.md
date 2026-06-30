# DeepSeek Agent Project

基于 DeepSeek API（OpenAI 兼容协议）构建的个人 AI 助手，支持工具调用、多轮对话持久化、子代理并行执行和语义缓存。用于Agent各类框架测试和学习

## 项目结构

```
Agent_Project/
├── agent.py          # Agent 主体：对话循环 + 工具调用 + 记忆恢复
├── tools.py          # 基础设施：ToolRegistry / SessionStore / SubagentPool / SemanticCache + 7 个工具函数
├── chat.py           # 纯对话 demo（无工具能力，快速验证的demo）
├── .env              # API 密钥（需要本地创建）
├── .gitignore
└── README.md
```

## 已实现的能力

### 核心循环（agent.py）

- **ReAct 循环**：模型输出→如有工具调用则执行→结果回喂→继续推理，最多 15 轮
- **流式输出**：最终回复逐 token 打印
- **会话持久化**：每轮对话写入 `agent_sessions.json`，支持跨轮次记忆恢复
- **语义缓存**：命中相似问题（余弦相似度 ≥ 0.85）时直接返回缓存回答，跳过 API 调用

### 工具系统（tools.py）

| 工具名 | 功能 |
|--------|------|
| `get_weather_tool` | 模拟天气查询（写死数据，仅测试用） |
| `calculator` | 安全数学表达式计算 |
| `read_file_tool` | 读取文件内容 |
| `write_file_tool` | 写入文件，自动创建父目录 |
| `shell_tool` | 执行 shell 命令（含危险命令黑名单拦截） |
| `search_files_tool` | 按通配符搜索文件名 |
| `grep_tool` | 在 .py/.ts/.md 文件中搜索文本 |
| `subagent_task` | 启动子代理执行独立任务 |

### 子代理（SubagentPool）

- 通过 `subagent_task` 工具暴露给模型，支持独立上下文执行
- `parallel_tasks()` 使用 `ThreadPoolExecutor` 并发运行多个子任务
- 子代理使用独立的 system prompt，与主 Agent 上下文隔离

### 基础设施

- **ToolRegistry**：工具注册、分发、Schema 导出
- **SessionStore**：基于 JSON 文件的会话持久化（append / load / delete）
- **SemanticCache**：基于词袋余弦相似度的语义缓存（5 分钟 TTL）

## 运行依赖

```bash
pip install openai==2.44.0 python-dotenv==1.2.2
```

仅此两个，其余均为 Python 3.12 标准库。

## 运行方式

```bash
pip install openai==2.44.0 python-dotenv==1.2.2
python agent.py
```

会话 ID 固定为 `"1"`，重启后自动恢复上次对话历史。

## 已知待改进项

- 语义缓存的 `_embed()` 使用空格分词，对中文无效（需要 `jieba` 或字符级方案）
- 无成本追踪（未记录每次 API 调用的 token 用量）
- 无 MCP 集成（工具需手写注册）
- 主 Agent 和子代理共享同一 API key 和模型配置
- 无生产级 HTTP 服务（当前为终端交互）
