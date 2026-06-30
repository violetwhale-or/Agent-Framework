"""
mcp_manager.py — 多 MCP 服务器管理器（同步版）

不再需要 asyncio / anyio。所有操作均为同步阻塞调用。
"""
import json
from typing import Any

from mcp_client import MCPClient


class MCPServerConfig:
    """描述一个 MCP 服务器的配置"""

    def __init__(self, name: str, command: str, args: list[str]):
        self.name = name
        self.command = command
        self.args = args


class MCPManager:
    """管理多个 MCP 服务器连接（同步，无 asyncio）"""

    def __init__(self):
        self._clients: dict[str, MCPClient] = {}
        self._server_names: dict[str, str] = {}  # tool_name -> server_name
        self._original_names: dict[str, str] = {}  # tool_name -> original_name

    def add_server(self, config: MCPServerConfig):
        """注册一个服务器配置（启动时才会连接）"""
        client = MCPClient(config.name, config.command, config.args)
        self._clients[config.name] = client

    def connect_all(self):
        """连接所有已注册的服务器"""
        for name, client in self._clients.items():
            try:
                client.connect()
                print(f"  ✅ MCP 服务器 [{name}] 已连接")
            except Exception as e:
                print(f"  ❌ MCP 服务器 [{name}] 连接失败: {e}")

    def disconnect_all(self):
        """断开所有服务器连接"""
        for name, client in self._clients.items():
            try:
                client.disconnect()
            except Exception:
                pass

    def get_all_tools(self) -> list[dict]:
        """从所有连接的服务器获取工具列表"""
        all_tools = []
        for name, client in self._clients.items():
            try:
                tools = client.list_tools()
                for t in tools:
                    prefixed_name = f"{name}_{t['name']}"
                    self._server_names[prefixed_name] = name
                    self._original_names[prefixed_name] = t['name']
                    all_tools.append({
                        "server": name,
                        "original_name": t['name'],
                        "name": prefixed_name,
                        "description": t['description'],
                        "inputSchema": t['inputSchema'],
                    })
            except Exception as e:
                print(f"  ❌ 获取 [{name}] 工具列表失败: {e}")
        return all_tools

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """在指定服务器上调用工具"""
        server = self._server_names.get(tool_name)
        if not server:
            return json.dumps({"error": f"未知工具: {tool_name}"})
        client = self._clients.get(server)
        if not client:
            return json.dumps({"error": f"未知 MCP 服务器: {server}"})
        try:
            return client.call_tool(
                self._original_names.get(tool_name, tool_name),
                arguments,
            )
        except Exception as e:
            return json.dumps({"error": str(e)})
