import os 
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ['DEEPSEEK_API_KEY'],
    base_url="https://api.deepseek.com",
)

messages = [{"role": "system", "content": "你是一个知识渊博且忠于规则限制会高效处理问题的AI助手。用中文回答。"}]

while True:
    user = input("\n> ")
    if user.lower() in ("quit", "exit", "q"):
        break
    messages.append({"role": "user", "content": user})

    stream = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.7,
        max_tokens=4096,
        stream=True,
    )
    print()
    full = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            full += chunk.choices[0].delta.content
    print()
    messages.append({"role": "assistant", "content": full})