"""
feishu_bot.py — 飞书机器人入口

通过飞书 WebSocket 长连接接收消息，转发给 Agent 处理。

用法：
  export FEISHU_APP_ID=cli_xxx
  export FEISHU_APP_SECRET=xxx
  python feishu_bot.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from agentv2 import Agent
from channels import FeishuChannel


def main():
    load_dotenv()

    app_id = os.environ.get("FEISHU_APP_ID")
    app_secret = os.environ.get("FEISHU_APP_SECRET")

    if not app_id or not app_secret:
        print("请设置环境变量 FEISHU_APP_ID 和 FEISHU_APP_SECRET")
        print("或写入 .env 文件：")
        print("  FEISHU_APP_ID=cli_xxx")
        print("  FEISHU_APP_SECRET=xxx")
        sys.exit(1)

    print("启动飞书机器人...")
    agent = Agent(max_turns=20)
    channel = FeishuChannel(agent, app_id, app_secret)

    try:
        channel.start()
    except KeyboardInterrupt:
        print("\n收到退出信号")
    finally:
        channel.stop()
        print("飞书机器人已停止")


if __name__ == "__main__":
    main()
