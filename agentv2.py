import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import Generator
from tools import ToolRegistry, SessionStore, SubagentPool, SemanticCache
from mcp_manager import MCPManager, MCPServerConfig


class Agent:
    def __init__(self, max_turns: int = 15):
        self.client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        self.max_turns = max_turns
        self.registry = ToolRegistry()
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

        # 连接 MCP 服务器并注册 MCP工具
        self.mcp = MCPManager()
        self._connect_mcp_servers()

    def run_stream(self, user_input: str, session_id = None) -> Generator[str, None, None]:
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
                                        "content":  msg.content or "", 
                                        "tool_calls": [
                                            {"id": tc.id, "type": "function",
                                            "function": {"name": tc.function.name, "arguments":
                                                        tc.function.arguments}}
                                            for tc in msg.tool_calls
                                        ]})

                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    yield f"  🔧 调用工具：{name}({json.dumps(args, ensure_ascii=False)})"
                    result = self.registry.dispatch(name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                    self.store.append(session_id=session_id, turn={"role": "tool", "content":  result, "tool_call_id": tc.id})
                continue

            else:
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

                self.store.append(session_id=session_id, turn={"role": "assistant", "content":  full})
                yield "[DONE]"
                return 

        return "达到最大轮次上限"
    
    def _system_prompt(self):
        return (
            "你是一个 AI 编程助手，类似 Claude Code。\n"
            "规则：\n"
            "- 每次行动前先思考是否需要工具\n"
            "- 用工具获取实际信息，不要猜测文件内容\n"
            "- 完成用户请求后直接回复，不要额外调用工具\n"
            "- 如果 shell 命令返回错误，读报错信息，不要凭空猜测"
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
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                        "function": {"name": tc.function.name, "arguments":
                                    tc.function.arguments}}
                        for tc in msg.tool_calls
                    ]
                })

                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    print(f"  🔧 调用工具：{name}({json.dumps(args, ensure_ascii=False)})")
                    result = self.registry.dispatch(name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                continue

            else:

                full = ""
                for chunk in self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    stream=True,
                ):
                    if chunk.choices[0].delta.content:
                        full += chunk.choices[0].delta.content

                messages.append({"role": "assistant", "content": full})

                return full

        return "达到最大轮次上限"
    
    def _connect_mcp_servers(self):
        """连接配置的 MCP 服务器，把工具注册到 registry"""
        # 配置 MCP 服务器 (这里需要新的MCP库只需要把下面增加一个新的add_server就可以了，需要的仓库地址需要在MCP官网寻找)
        allowed_dir = os.path.abspath(".")
        self.mcp.add_server(MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", allowed_dir],
        ))

        self.mcp.add_server(MCPServerConfig(
            name="notion",
            command="npx",
            args=["-y", "@notionhq/notion-mcp-server"],
        ))

        # 连接所有服务器
        self.mcp.connect_all()

        # 从所有服务器上拉取工具， 注册到 registry
        mcp_tools = self.mcp.get_all_tools()
        for tool in mcp_tools:
            # 每个 MCP 工具包成一个本地调用的包
            server_name = tool["server"]
            original_name = tool["original_name"]
            tool_name = tool["name"]

            def make_dispatch(srv, tname):
                def dispatch_fn(**kwargs):
                    return self.mcp.call_tool(srv, tname, kwargs)
                dispatch_fn.__name__ = tname
                return dispatch_fn
            
            self.registry.register(
                tool_name,
                make_dispatch(server_name, original_name),
                tool["description"],
                tool["inputSchema"],
            )
            print(f"  📦 MCP 工具注册: {tool_name}")

        print(f"  ✅ 共注册 {len(mcp_tools)} 个 MCP 工具")
    
    def close(self):
        """主动关闭——断开 MCP 连接"""
        if hasattr(self, 'mcp'):
            try:
                self.mcp.disconnect_all()
            except Exception:
                pass

    def __del__(self):
        """析构时不做任何异步操作（Python 退出时事件循环不可用）"""
        pass

if __name__ == "__main__":
    load_dotenv()
    agent = Agent(max_turns=15)

    while True:
        user_input = input("\n> ")
        if user_input.lower() in ("quit", "exit", "q"):
            agent.close()
            break
        for token in agent.run_stream(user_input=user_input, session_id= "1"):
            print(token, end="", flush=True)