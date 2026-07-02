import json 
import datetime
import math
import time
import subprocess
import pathlib
import glob
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from rag_tool import rag_query

class ToolRegistry:
    def __init__(self):
        self._tools:dict[str, dict] = {}

    def register(self, name: str, fn: Callable, description: str, parameters: dict):
        self._tools[name] = {
            "fn": fn,
            "definition": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                }
            }
        }
    
    def dispatch(self, name: str, args: dict) -> str:
        tool = self._tools.get(name)
        if not tool:
            return json.dumps({"error": f"unknown tool: {name}"})
        try:
            result = tool["fn"](**args)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})
        
    def schemas(self) -> list[dict]:
        return [t["definition"] for t in self._tools.values()]
    
    def list_tools(self) -> list[str]:
        return list(self._tools.keys())
    

@dataclass
class Turn:
    role: str    # "user" | "assistant" | "tool"
    content: str
    tool_name: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class SessionStore:
    '''实现对话存储'''
    def __init__(self, path: str = "sessions.json"):
        self.path = path
        self._sessions: dict[str, list[dict]] = {}
        self._load()

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                self._sessions = json.load(f)
        except FileNotFoundError:
            self._sessions = {}

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._sessions, f, indent=2, ensure_ascii=False)

    def append(self, session_id: str, turn: dict):
        self._sessions.setdefault(session_id, []).append(turn)
        self._save()

    def load(self, session_id: str) -> list[dict]:
        return list(self._sessions.get(session_id, []))

    def list_sessions(self) -> list[str]:
        return sorted(self._sessions)

    def delete(self, session_id: str):
        self._sessions.pop(session_id, None)
        self._save()    


class SubagentPool:
    """管理子代理的创建和结果聚合"""

    def __init__(self, parent_agent):
        self.parent = parent_agent
        self._counter = 0

    def spawn(self, task: str, tools: Optional[list[str]] = None) -> dict:
        """创建子代理，用独立 context 执行任务"""
        self._counter += 1
        sub_id = f"sub-{self._counter:03d}"

        result = self.parent._run_loop(
            system_prompt="你是一个聚焦的子代理，专注完成指定任务后直接返回结果。",
            user_message=task,
        )
        return {"sub_id": sub_id, "result": result}

    def parallel_tasks(self, tasks: list[str]) -> list[dict]:
        """并行执行多个子任务"""
        with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
            return list(ex.map(self.spawn, tasks))
        

class SemanticCache:
    def __init__(self, threshold = 0.85, ttl_seconds = 300):
        self.entries = []   # 每个元素: {"query": str, "embedding": dict, "response": str, "timestamp": float}
        self.threshold = threshold
        self.ttl = ttl_seconds

    def _embed(self, text: str) -> dict:
        words = text.lower().split()
        vec = {}
        for w in words:
            vec[w] = vec.get(w, 0) + 1
        norm = math.sqrt(sum(v*v for v in vec.values()))
        return {k: v/norm for k, v in vec.items()} if norm > 0 else {}
    
    def _cosine(self, a: dict, b: dict) -> float:
        keys = set(a) | set(b)
        return sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    
    def get(self, query: str) -> str | None:
        q_emb = self._embed(query)
        now = time.time()
        for e in self.entries:
            if now - e["timestamp"] > self.ttl:
                continue
            if self._cosine(q_emb, e["embedding"]) >= self.threshold:
                return e["response"]
        return None
    
    def put(self, query: str, response: str):
        self.entries.append({
            "query": query,
            "embedding": self._embed(query),
            "response": response,
            "timestamp": time.time(),
        })
        if len(self.entries) > 500:
            self.entries.pop(0)


def read_file_tool(path: str) -> dict:
    """读取文件内容"""

    p = pathlib.Path(path)
    if not p.exists():
        return {"error": f"文件不存在: {path}"}
    try:
        content = p.read_text(errors="replace")
        return {"path": str(p), "content": content, "size": len(content)}
    except Exception as e:
        return {"error": str(e)}

def write_file_tool(path: str, content: str) -> dict:
    """写入文件"""

    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return {"path": str(p), "bytes_written": len(content)}

def shell_tool(command: str) -> dict:
    """运行 shell 命令（带安全限制）"""

    dangerous = ["rm -rf /", "sudo ", "mkfs", "dd if="]
    for d in dangerous:
        if d in command.lower():
            return {"error": f"危险命令被拦截: {d}"}
    try:
        result = subprocess.run(command, shell=True, capture_output=True,
                                text=True, timeout=30)
        return {"stdout": result.stdout[-4000:], 
                "stderr": result.stderr[-2000:],
                "returncode": result.returncode,
            }
    except subprocess.TimeoutExpired:
        return {"error": "命令执行超时"}

def calculator(expression: str) -> dict:
    '''数学计算工具'''

    allowed = set("0123456789+-*/(). ")
    if not set(expression).issubset(allowed):
        return {"error": "表达式含非法字符"}
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}

def get_weather_tool(city: str) -> Dict[str, Any]:
    # 模拟数据（仅供测试）
    return {
        "city": city,
        "temperature": 22.5,
        "condition": "晴",
        "humidity": 65,
        "wind_speed": 12.0
    }

def search_files_tool(pattern: str, path: str = ".") -> dict:
    '''搜索文件名，输入pattern和path字符串，返回是字典格式'''
    results = glob.glob(f"{path}/**/{pattern}", recursive=True)[:50]
    return {"pattern": pattern, "matches": results, "count": len(results)}

def grep_tool(query: str, path: str = ".") -> dict:
    '''搜索文件内容，输入query和path为字符串，返回是字典格式'''
    result = subprocess.run(
        ["grep", "-rn", "--include=*.py", "--include=*.ts",
         "--include=*.md", query, path],
        capture_output=True, text=True, timeout=10,
    )
    lines = result.stdout.splitlines()[:30]
    return {"query": query, "matches": lines, "count": len(lines)}


def build_default_registry() -> ToolRegistry:
    '''创建并返回默认的工具注册表（所有预置工具已注册）'''

    r = ToolRegistry()

    r.register(
        "read_file_tool",
        read_file_tool,
        "读取文件内容，返回文件内容字符，在路径不存在或是没有正确打开时会返回 error",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"}
            },
            "required": ["path"]
        }
    )

    r.register(
        "write_file_tool",
        write_file_tool,
        "将内容写入指定文件，自动创建父目录。返回包含文件路径(path)和写入字节数(bytes_written)的字典。",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要写入的目标文件路径"},
                "content": {"type": "string", "description": "需要写入文件的文本内容"}
            },
            "required": ["path", "content"]
        }
    )

    r.register(
        "shell_tool",
        shell_tool,
        "执行系统 Shell 命令（带安全拦截），返回包含 stdout、stderr 和 returncode 的字典，超时或拦截时返回 error。",
        {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "需要执行的完整 Shell 命令字符串"}
            },
            "required": ["command"]
        }
    )

    r.register(
        "calculator",
        calculator,
        "安全计算数学表达式（仅允许数字和 + - * / ( ) . 空格），返回表达式和计算结果，非法字符返回 error。",
        {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式，例如 '2 + 3 * 4'"}
            },
            "required": ["expression"]
        }
    )

    r.register(
        "get_weather_tool",                    
        get_weather_tool,                    
        "获取指定城市的实时天气信息（温度、天气状况、湿度、风速）。"  
        "返回字典包含 city, temperature(°C), condition, humidity(%), wind_speed(km/h)。"
        "如果请求失败，返回包含 error 字段的字典。",
        {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，支持中文或英文，例如 '上海' 或 'Shanghai'"
                }
            },
            "required": ["city"]              
        }
    )

    r.register(
        "search_files_tool",                    
        search_files_tool,                    
        "搜索指定文件名的文件"  
        "返回字典包含 pattern（搜索模式）、matches（匹配的文件路径列表，最多50条）、count（匹配总数）。"
        "支持通配符，例如 '*.txt' 或 'data/*.csv'。",
        {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "文件匹配模式，支持通配符 * 和 ?，例如 '*.py' 或 '**/test_*.json'"
                },
                "path": {
                    "type": "string",
                    "description": "搜索的根目录路径，默认为当前目录（可以不传）"
                },
            },
            "required": ["pattern"]              
        }
    )

    r.register(
        "grep_tool",
        grep_tool,
        "在指定目录（默认当前目录）下递归搜索 .py、.ts、.md 文件中包含指定文本的行，返回匹配行列表（最多30行）。"
        "返回字典包含 query（搜索文本）、matches（匹配行列表）、count（匹配行数）。",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要搜索的文本字符串"
                },
                "path": {
                    "type": "string",
                    "description": "搜索的根目录路径，默认为当前目录（可以不传）"
                }
            },
            "required": ["query"]   # 只有 query 是必填，因为 path 有默认值
        }
    )

    r.register(
        "rag_query",
        rag_query,
        "从知识库中检索与问题语义相似的原始文本段落（不调 LLM 生成）。"
        "返回多段文本，由主 LLM 自行判断是否采用。未匹配时返回「未找到相关信息」。",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要搜索的问题字符串"
                },
            },
            "required": ["query"]   # 只有 query 是必填，因为 path 有默认值
        }
    )

    return r
