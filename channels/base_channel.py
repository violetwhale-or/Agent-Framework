"""
base_channel.py — 通道抽象基类

所有外部消息通道（终端、Web、飞书、钉钉等）统一实现此接口。
"""

from abc import ABC, abstractmethod
from typing import Generator


class BaseChannel(ABC):
    """消息通道抽象基类"""

    @abstractmethod
    def send_message(self, target: str, text: str):
        """发送消息到指定目标"""
        ...

    @abstractmethod
    def send_stream(self, target: str, tokens: Generator[str, None, None]):
        """流式发送消息（通道支持流式时实现，不支持则退化为非流式）"""
        ...

    @abstractmethod
    def start(self):
        """启动通道服务（阻塞或后台运行）"""
        ...

    def stop(self):
        """停止通道服务"""
        pass
