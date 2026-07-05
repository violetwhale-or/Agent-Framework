FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖（chromadb 的 SQLite 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 持久化知识库和会话数据
VOLUME ["/app/rag_data", "/app/agent_sessions.json"]

EXPOSE 8000

CMD ["python", "server.py"]

