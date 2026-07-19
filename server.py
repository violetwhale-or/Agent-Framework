import json
import os
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv
from agentv2 import Agent
from agent_config import config


# 让相对路径相对于 server.py 所在目录，而不是运行目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv()

app = FastAPI(title="Deepseek Agent")

# CORS——允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = Agent(max_turns=config.llm.max_turns)

SESSIONS_FILE = os.path.join(BASE_DIR, "sessions_index.json")

def _load_sessions() -> dict[str, str]:
    try:
        with open(SESSIONS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}

def _save_sessions(s: dict[str, str]):
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False)

sessions: dict[str, str] = _load_sessions()

@app.get("/")
async def root():
    """返回首页 HTML"""
    with open(os.path.join(BASE_DIR, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/sessions")
async def list_sessions():
    """返回所有会话列表（含首条消息预览）"""
    if not sessions:
        sid = str(uuid.uuid4())[:8]
        sessions[sid] = "新对话"
        _save_sessions(sessions)
    result = []
    for sid, name in sessions.items():
        history = agent.memory.short.load(sid)
        preview = ""
        for msg in history:
            if msg.get("role") == "user":
                preview = msg["content"][:30]
                break
        result.append({"id": sid, "name": name, "preview": preview})
    return {"sessions": result}


@app.post("/api/sessions")
async def create_session():
    """创建新会话"""
    sid = str(uuid.uuid4())[:8]
    sessions[sid] = f"对话 {len(sessions) + 1}"
    _save_sessions(sessions)
    return {"session_id": sid, "name": sessions[sid]}


@app.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """返回指定会话的历史消息列表"""
    history = agent.memory.short.load(session_id)
    return {"messages": history}


@app.post("/chat/{session_id}")
async def chat(session_id: str, request: Request):
    """
    SSE 流式聊天接口。
    浏览器调这个接口，拿到 EventSource 流，逐 token 显示。
    """
    body = await request.json()
    user_message = body.get("message", "")

    if session_id not in sessions:
        sessions[session_id] = f"对话 {len(sessions) + 1}"
        _save_sessions(sessions)

    async def event_generator():
        """把 agent.run_stream() 的 yield 包装成 SSE 事件

        事件类型：
        - thinking: [tool]/[tokens] 思考过程
        - token: 模型输出正文
        - done: 回复结束
        - error: 错误信息
        """
        thinking_buffer = ""
        text_buffer = ""
        try:
            for token in agent.run_stream(
                user_input=user_message,
                session_id=session_id,
            ):
                if await request.is_disconnected():
                    break

                if token == "[DONE]":
                    # 发完剩余 buffer
                    if text_buffer:
                        yield {"event": "token", "data": json.dumps({"text": text_buffer})}
                    if thinking_buffer:
                        yield {"event": "thinking", "data": json.dumps({"text": thinking_buffer})}
                    yield {"event": "done", "data": ""}
                    break

                if token.startswith("[tool]") or token.startswith("[tokens]"):
                    # 思考过程：单独用 thinking 事件推送
                    thinking_buffer += token + "\n"
                    if text_buffer:
                        yield {"event": "token", "data": json.dumps({"text": text_buffer})}
                        text_buffer = ""
                    yield {"event": "thinking", "data": json.dumps({"text": token})}
                elif token.startswith("[ERROR]"):
                    yield {"event": "error", "data": json.dumps({"error": token})}
                else:
                    # 正文文本
                    text_buffer += token
                    if any(p in token for p in "。！？\n；\n"):
                        yield {"event": "token", "data": json.dumps({"text": text_buffer})}
                        text_buffer = ""
                    elif len(text_buffer) >= 50:
                        yield {"event": "token", "data": json.dumps({"text": text_buffer})}
                        text_buffer = ""
        except Exception as e:
            if text_buffer:
                yield {"event": "token", "data": json.dumps({"text": text_buffer})}
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.server.host, port=config.server.port)