"""
mcp_client.py — 同步 subprocess 实现，零异步依赖

MCP 协议基于 JSON-RPC 2.0 over stdio：
  客户端 → stdin:  {"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}
  服务器 ← stdout: {"jsonrpc":"2.0","id":1,"result":{"tools":[...]}}
"""
import json
import subprocess
import shutil
from typing import Any


class MCPClient:
    """管理一个 MCP 服务器连接（同步 subprocess，无 asyncio）"""

    def __init__(self, server_name: str, command: str, args: list[str]):
        self.server_name = server_name
        self.command = command
        self.args = args
        self._process: subprocess.Popen | None = None
        self._request_id = 0

    def connect(self):
        """启动子进程，完成 MCP 握手"""
        executable = shutil.which(self.command)
        if executable is None:
            raise FileNotFoundError(
                f"找不到可执行文件 '{self.command}'。"
                f"请确认已安装，或使用完整路径。"
            )
        self._process = subprocess.Popen(
            [executable] + self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        # MCP 初始化握手
        result = self._request("initialize", {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "my-agent", "version": "0.1.0"},
        })
        if "result" not in result:
            raise RuntimeError(f"MCP 握手失败: {result}")
        # 发送 initialized 通知
        self._notify("notifications/initialized", {})

    def disconnect(self):
        """关闭子进程"""
        if self._process:
            try:
                self._process.stdin.close()
            except Exception:
                pass
            try:
                self._process.terminate()
            except Exception:
                pass
            self._process = None

    def list_tools(self) -> list[dict]:
        """获取工具列表"""
        result = self._request("tools/list", {})
        tools_data = result.get("result", {}).get("tools", [])
        return [
            {"name": t["name"], "description": t.get("description", ""),
             "inputSchema": t.get("inputSchema", {})}
            for t in tools_data
        ]

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """调用工具"""
        result = self._request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        content = result.get("result", {}).get("content", [])
        parts = []
        for c in content:
            if c.get("type") == "text":
                parts.append(c.get("text", ""))
            elif c.get("type") == "resource":
                parts.append(str(c.get("resource", "")))
            else:
                parts.append(str(c))
        return "\n".join(parts)

    def _request(self, method: str, params: dict) -> dict:
        """发送 JSON-RPC 请求，等待响应"""
        if not self._process or not self._process.stdin:
            raise ConnectionError(f"MCP 服务器 [{self.server_name}] 未连接")
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        line = json.dumps(request, ensure_ascii=False) + "\n"
        self._process.stdin.write(line)
        self._process.stdin.flush()

        response_line = self._process.stdout.readline()
        if not response_line:
            raise ConnectionError("MCP 服务器连接已关闭")
        return json.loads(response_line)

    def _notify(self, method: str, params: dict):
        """发送 JSON-RPC 通知（不需要响应）"""
        if not self._process or not self._process.stdin:
            return
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        line = json.dumps(notification, ensure_ascii=False) + "\n"
        self._process.stdin.write(line)
        self._process.stdin.flush()
