import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from tools import build_default_registry, SessionStore, SubagentPool, SemanticCache


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

    def run(self, user_input: str, session_id = None) -> str:
        messages = [{"role": "system", "content": self._system_prompt()}]

        if session_id:
            history = self.store.load(str(session_id))
            messages.extend(history) 

        if user_input.lower() in ("quit", "exit", "q"):
            return "Quit"


        cached = self.cache.get(user_input)
        if cached:
            print(cached)
            self.store.append(session_id, {"role": "assistant", "content": cached})
            return cached

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
                    print(f"  🔧 调用工具：{name}({json.dumps(args, ensure_ascii=False)})")
                    result = self.registry.dispatch(name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                    self.store.append(session_id=session_id, turn={"role": "tool", "content":  result, "tool_call_id": tc.id})
                continue

            else:
                print()
                full = ""
                for chunk in self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    stream=True,
                ):
                    if chunk.choices[0].delta.content:
                        print(chunk.choices[0].delta.content, end="", flush=True)
                        full += chunk.choices[0].delta.content
                print()
                messages.append({"role": "assistant", "content": full})

                self.store.append(session_id=session_id, turn={"role": "assistant", "content":  full})
                return full

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
                print()
                full = ""
                for chunk in self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    stream=True,
                ):
                    if chunk.choices[0].delta.content:
                        print(chunk.choices[0].delta.content, end="", flush=True)
                        full += chunk.choices[0].delta.content
                print()
                messages.append({"role": "assistant", "content": full})

                return full

        return "达到最大轮次上限"
    

if __name__ == "__main__":
    load_dotenv()
    agent = Agent(max_turns=15)

    while True:
        user_input = input("\n> ")

        res = agent.run(user_input=user_input, session_id= "1")
        if res == "Quit":
            break