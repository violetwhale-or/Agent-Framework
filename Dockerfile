FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖（chromadb 的 SQLite 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 安装 Node.js 并全局安装 MCP 服务器包
RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs npm \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @modelcontextprotocol/server-filesystem @notionhq/notion-mcp-server

# 复制项目代码
COPY . .

# 持久化知识库
VOLUME ["/app/rag_data"]

EXPOSE 8000

CMD ["python", "server.py"]
