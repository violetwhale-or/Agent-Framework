import json
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv
from agentv2 import Agent
import asyncio

load_dotenv()

app = FastAPI(title="Deepseek Agent")

# CORS——允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = Agent(max_turns=15)

sessions: dict[str, str] = {}

@app.get("/")
async def root():
    """返回首页 HTML"""
    with open("index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/sessions")
async def list_sessions():
    """返回所有会话列表"""
    if not sessions:
        # 默认创建一个会话
        sid = str(uuid.uuid4())[:8]
        sessions[sid] = "新对话"
    return {"sessions": [{"id": k, "name": v} for k, v in sessions.items()]}


@app.post("/api/sessions")
async def create_session():
    """创建新会话"""
    sid = str(uuid.uuid4())[:8]
    sessions[sid] = f"对话 {len(sessions) + 1}"
    return {"session_id": sid, "name": sessions[sid]}


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

    async def event_generator():
        """把 agent.run_stream() 的 yield 包装成 SSE 事件"""
        buffer = ""  # 攒一段再发
        try:
            for token in agent.run_stream(
                user_input=user_message,
                session_id=session_id,
            ):
                if await request.is_disconnected():
                    break
                if token == "[DONE]":
                    # 把最后攒的也发出去
                    if buffer:
                        yield {"event": "token", "data": json.dumps({"text": buffer})}
                        buffer = ""
                    yield {"event": "done", "data": ""}
                else:
                    buffer += token
                    # 句子结束标点或超过 50 字则 flush
                    if any(p in token for p in "。！？\n；\n"):
                        yield {"event": "token", "data": json.dumps({"text": buffer})}
                        buffer = ""
                    elif len(buffer) >= 50:
                        yield {"event": "token", "data": json.dumps({"text": buffer})}
                        buffer = ""
        except Exception as e:
            if buffer:
                yield {"event": "token", "data": json.dumps({"text": buffer})}
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)