import os
import json
import shutil
from openai import OpenAI
from dotenv import load_dotenv
from typing import Generator
from tools import build_default_registry, SessionStore, SubagentPool, SemanticCache
from mcp_manager import MCPManager, MCPServerConfig


class Agent:
    def __init__(self, max_turns: int = 15):
        self.client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        self.max_turns = max_turns
        self.registry = build_default_registry()
        self.store = SessionStore("agent_sessions.json")
        self.subagent_pool = SubagentPool(self)
        self.registry.register(
            "subagent_task",
            self.subagent_pool.spawn,
            "创建一个子代理来执行独立任务",
            {
                "type": "object",
                "properties":{
                    "task":{"type": "string", "description": "子代理的独立任务内容"}
                },
                "required": ['task']
            }
        )
        self.cache = SemanticCache(threshold=0.85, ttl_seconds=300)

        self.mcp = MCPManager()
        self._mcp_ready = False
        self._mcp_enabled = False
        self._mcp_tool_names: list[str] = []
        self._mcp_tool_defs: list[dict] = []

    def _connect_mcp_servers(self):
        allowed_dir = os.path.abspath(".")

        def _resolve(cmd: str, pkg: str, args: list[str]):
            if shutil.which(cmd):
                return cmd, args
            return "npx", ["-y", pkg] + args

        fs_cmd, fs_args = _resolve("mcp-server-filesystem",
            "@modelcontextprotocol/server-filesystem", [allowed_dir])
        self.mcp.add_server(MCPServerConfig("filesystem", fs_cmd, fs_args))

        nn_cmd, nn_args = _resolve("notion-mcp-server",
            "@notionhq/notion-mcp-server", [])
        self.mcp.add_server(MCPServerConfig("notion", nn_cmd, nn_args))

        self.mcp.connect_all()

        self._mcp_tool_names = []
        self._mcp_tool_defs = []
        for tool in self.mcp.get_all_tools():
            tool_name = tool["name"]
            self._mcp_tool_names.append(tool_name)
            def make_dispatch(srv, tname):
                def dispatch_fn(**kwargs):
                    return self.mcp.call_tool(srv, tname, kwargs)
                dispatch_fn.__name__ = tname
                return dispatch_fn
            defn = {
                "name": tool_name,
                "fn": make_dispatch(tool["server"], tool["original_name"]),
                "description": tool["description"],
                "inputSchema": tool["inputSchema"],
            }
            self._mcp_tool_defs.append(defn)
            self.registry.register(
                tool_name, defn["fn"],
                defn["description"], defn["inputSchema"],
            )

        self._mcp_enabled = True
        print(f"[MCP] {len(self._mcp_tool_names)} tools registered")

    def _enable_mcp(self):
        if not self._mcp_ready:
            self._mcp_ready = True
            self._connect_mcp_servers()
            return
        if self._mcp_enabled:
            return
        for t in self._mcp_tool_defs:
            self.registry.register(t["name"], t["fn"], t["description"], t["inputSchema"])
        self._mcp_enabled = True

    def _disable_mcp(self):
        for name in self._mcp_tool_names:
            self.registry.remove(name)
        self._mcp_enabled = False

    def run_stream(self, user_input: str, session_id = None) -> Generator[str, None, None]:
        cmd = user_input.strip().lower()
        if cmd == "/mcp on":
            self._enable_mcp()
            yield "MCP 工具已启用。当前 MCP 工具：\n" + "\n".join(f"- {n}" for n in self._mcp_tool_names)
            return
        if cmd == "/mcp off":
            self._disable_mcp()
            yield "MCP 工具已禁用，切换为本地工具。输入 /mcp on 重新启用。"
            return
        if cmd in ("/help", "/h"):
            yield "可用命令：\n  /mcp on  - 启用 MCP 工具\n  /mcp off - 禁用 MCP 工具\n  /help    - 显示此帮助"
            return

        messages = [{"role": "system", "content": self._system_prompt()}]

        if session_id:
            history = self.store.load(str(session_id))
            messages.extend(history)

        if user_input.lower() in ("quit", "exit", "q"):
            return "Quit"

        cached = self.cache.get(user_input)
        if cached:
            yield cached
            self.store.append(session_id, {"role": "assistant", "content": cached})
            return

        messages.append({"role": "user", "content": user_input})
        self.store.append(session_id=session_id, turn={"role": "user", "content": user_input})

        for turn in range(self.max_turns):
            tc = "none" if turn == self.max_turns - 1 else "auto"

            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=self.registry.schemas(),
                tool_choice=tc,
                temperature=0.3,
            )

            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                        "function": {"name": tc.function.name, "arguments":
                                    tc.function.arguments}}
                        for tc in msg.tool_calls
                    ]
                })
                self.store.append(session_id=session_id,
                                  turn={"role": "assistant",
                                        "content": msg.content or "",
                                        "tool_calls": [
                                            {"id": tc.id, "type": "function",
                                            "function": {"name": tc.function.name, "arguments":
                                                        tc.function.arguments}}
                                            for tc in msg.tool_calls
                                        ]})

                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    yield f"[tool] {name}({json.dumps(args, ensure_ascii=False)})"
                    result = self.registry.dispatch(name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                    self.store.append(session_id=session_id, turn={"role": "tool", "content": result, "tool_call_id": tc.id})
                continue

            full = ""
            for chunk in self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=True,
            ):
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    full += chunk.choices[0].delta.content

            messages.append({"role": "assistant", "content": full})
            self.store.append(session_id=session_id, turn={"role": "assistant", "content": full})
            yield "[DONE]"
            return

        yield "[ERROR] max turns reached"

    def _system_prompt(self):
        return (
            "你是一个 AI 编程助手。\n"
            "规则：\n"
            "- 每次行动前先思考是否需要工具\n"
            "- 用工具获取实际信息，不要猜测文件内容\n"
            "- 完成用户请求后直接回复，不要额外调用工具\n"
            "- 如果 shell 命令返回错误，读报错信息，不要凭空猜测\n"
            "- 调用 rag_query 工具后，检查返回内容是否与问题相关。\n"
            "  如果相关则基于知识回答；如果不相关则用自己的知识回答。\n"
        )

    def _run_loop(self, user_message: str, system_prompt: str = None) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": user_message})

        for _ in range(self.max_turns):
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=self.registry.schemas(),
                tool_choice="auto",
                temperature=0.3,
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append({
                    "role": "assistant", "content": msg.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ]
                })
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    result = self.registry.dispatch(tc.function.name, args)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                continue

            full = ""
            for chunk in self.client.chat.completions.create(
                model="deepseek-chat", messages=messages, stream=True,
            ):
                if chunk.choices[0].delta.content:
                    full += chunk.choices[0].delta.content
            return full

        return "达到最大轮次上限"

    def close(self):
        if hasattr(self, 'mcp'):
            try:
                self.mcp.disconnect_all()
            except Exception:
                pass

    def __del__(self):
        pass


if __name__ == "__main__":
    load_dotenv()
    agent = Agent(max_turns=20)

    while True:
        user_input = input("\n> ")
        if user_input.lower() in ("quit", "exit", "q"):
            agent.close()
            break
        for token in agent.run_stream(user_input=user_input, session_id="1"):
            print(token, end="", flush=True)
