"""
feishu_channel.py — 飞书适配器（长连接 WebSocket 模式）

依赖：pip install lark-oapi

用法：
  feishu = FeishuChannel(agent)
  feishu.start()
"""

import os
import json
import threading
from typing import Generator, Optional
from channels.base_channel import BaseChannel

try:
    import lark_oapi as lark
    from lark_oapi.api.im.v1 import *
    from lark_oapi.api.im.v1.model.create_message_request_body import CreateMessageRequestBody
    from lark_oapi.ws import Client as WSClient
    from lark_oapi.event.dispatcher_handler import EventDispatcherHandler
except ImportError:
    lark = None


class FeishuChannel(BaseChannel):
    """飞书适配器，通过 WebSocket 长连接与飞书通信"""

    def __init__(self, agent, app_id: str = None, app_secret: str = None):
        if lark is None:
            raise ImportError("缺少 lark-oapi 库，请执行: pip install lark-oapi")

        self.agent = agent
        self.app_id = app_id or os.environ.get("FEISHU_APP_ID", "")
        self.app_secret = app_secret or os.environ.get("FEISHU_APP_SECRET", "")
        self._ws_client: Optional[WSClient] = None
        self._running = False
        self._processed_msg_ids: set = set()

    def _text_to_post_content(self, text: str) -> str:
        """将纯文本转换为飞书富文本 post 格式的 content JSON

        飞书富文本格式说明：
        - 每行是一个独立的段落（element 数组）
        - 段落内支持多种 tag：text（文本）、a（链接）、code_block（代码块）
        """
        lines = text.split("\n")
        paragraphs = []
        in_code_block = False
        code_lines = []

        for line in lines:
            # 检测代码块开始/结束
            if line.strip().startswith("```"):
                if in_code_block:
                    # 结束代码块
                    code_text = "\n".join(code_lines)
                    paragraphs.append([
                        {"tag": "code_block", "text": code_text, "language": "plaintext"}
                    ])
                    code_lines = []
                    in_code_block = False
                else:
                    in_code_block = True
                continue

            if in_code_block:
                code_lines.append(line)
                continue

            # 普通行：构建行内元素
            elements = self._parse_inline(line)
            paragraphs.append(elements)

        # 代码块未闭合时也提交
        if code_lines:
            code_text = "\n".join(code_lines)
            paragraphs.append([
                {"tag": "code_block", "text": code_text, "language": "plaintext"}
            ])

        post = {"zh_cn": {"content": paragraphs}}
        return json.dumps(post, ensure_ascii=False)

    def _parse_inline(self, line: str) -> list[dict]:
        """解析行内标记，返回 element 列表

        支持：
        - **粗体** → tag=text, style=["bold"]
        - `行内代码` → tag=text（保留反引号）
        - 纯文本 → tag=text
        """
        import re
        elements = []
        pos = 0

        # 按顺序匹配：代码块内联 > 粗体 > 普通文本
        pattern = re.compile(r"(`[^`]+`)|(\*\*(.+?)\*\*)")
        for m in pattern.finditer(line):
            start, end = m.start(), m.end()

            # 匹配前的纯文本
            if start > pos:
                elements.append({"tag": "text", "text": line[pos:start]})

            if m.group(1):  # 行内代码 `code` → 用文本展示（飞书 post 不支持 inline_code tag）
                code_text = m.group(1)[1:-1]
                elements.append({"tag": "text", "text": f"`{code_text}`"})
            elif m.group(2):  # 粗体 **text**
                bold_text = m.group(3)
                elements.append({"tag": "text", "text": bold_text, "style": ["bold"]})

            pos = end

        # 剩余纯文本
        if pos < len(line):
            elements.append({"tag": "text", "text": line[pos:]})

        return elements if elements else [{"tag": "text", "text": line}]

    def send_message(self, target: str, text: str, rich: bool = True):
        """发送飞书消息

        Args:
            target: 接收者 ID
            text: 消息文本
            rich: True=富文本(post格式), False=纯文本(text格式)
        """
        if not target:
            return
        client = lark.Client.builder() \
            .app_id(self.app_id) \
            .app_secret(self.app_secret) \
            .build()

        # 判断 ID 类型和目标
        id_type = "open_id" if target.startswith("ou_") else "chat_id"

        if rich:
            content = self._text_to_post_content(text)
            msg_type = "post"
        else:
            content = json.dumps({"text": text})
            msg_type = "text"

        body = CreateMessageRequestBody.builder() \
            .receive_id(target) \
            .msg_type(msg_type) \
            .content(content) \
            .build()

        request = CreateMessageRequest.builder() \
            .receive_id_type(id_type) \
            .request_body(body) \
            .build()

        client.im.v1.message.create(request)

    def send_stream(self, target: str, tokens: Generator[str, None, None]):
        """飞书不支持流式输出，收集后一次性发送"""
        full = ""
        for token in tokens:
            full += token
        clean = full.replace("[DONE]", "").strip()
        if clean:
            self.send_message(target, clean)

    def _handle_message(self, data):
        """处理飞书消息事件（在子线程中执行，不阻塞 WebSocket 事件循环）"""
        # 立即在新线程中处理，释放事件循环
        import threading
        threading.Thread(target=self._process_message, args=(data,), daemon=True).start()

    def _process_message(self, data):
        """实际的飞书消息处理逻辑"""
        try:
            event = data.event
            message = event.message
            msg_type = message.message_type
            chat_type = message.chat_type

            if msg_type != "text":
                return

            # 去重：避免飞书 WebSocket 重复推送同一条消息
            msg_id = message.message_id
            if msg_id in self._processed_msg_ids:
                return
            self._processed_msg_ids.add(msg_id)

            content = json.loads(message.content)
            user_input = content.get("text", "").strip()
            if not user_input:
                return

            sender = message.chat_id if chat_type == "group" else event.sender.sender_id.open_id

            print(f"[feishu] 收到消息: {user_input[:60]}... (chat_type={chat_type}, sender={sender[:20]}...)")

            print(f"[feishu] 开始调用 Agent...")
            collected = ""
            for token in self.agent.run_stream(
                user_input=user_input,
                session_id=f"feishu_{sender}",
            ):
                if token == "[DONE]":
                    print(f"[feishu] 收到 [DONE]，collected={len(collected)} 字")
                    break
                if not token.startswith("[tool]") and not token.startswith("[tokens]") and not token.startswith("[ERROR]"):
                    collected += token
            print(f"[feishu] Agent 循环结束")

            if collected.strip():
                self.send_message(sender, collected.strip())
                print(f"[feishu] 回复已发送 ({len(collected)} 字)")
            else:
                print(f"[feishu] 回复为空, collected=[{collected[:100]}]")
        except Exception as e:
            import traceback
            print(f"[feishu] 处理消息异常: {e}")
            traceback.print_exc()

    def start(self):
        """启动飞书 WebSocket 长连接"""
        print("[feishu] 连接飞书 WebSocket...")

        # 注册消息事件处理器
        handler = EventDispatcherHandler.builder("", "") \
            .register_p2_im_message_receive_v1(self._handle_message) \
            .build()

        # 创建 WebSocket 客户端
        self._ws_client = WSClient(
            app_id=self.app_id,
            app_secret=self.app_secret,
            event_handler=handler,
        )

        self._running = True
        print("[feishu] 连接成功，等待消息...")
        self._ws_client.start()

    def stop(self):
        """断开飞书连接"""
        self._running = False
        print("[feishu] 已断开")
