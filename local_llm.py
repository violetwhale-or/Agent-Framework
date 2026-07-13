"""
local_llm.py — 本地小模型封装（Qwen2.5-1.5B-Instruct 4-bit）

能力：
  - classify(): 任务分类，判断是否需要工具
  - summarize(): 对话摘要（淘汰轮次时使用）

加载策略：
  - 首次调用时懒加载，不阻塞主进程启动
  - 4-bit 量化，显存 ~750MB，CPU 也可运行
  - 模型下载到本地后自动缓存，后续可离线加载

首次下载：
  export HF_ENDPOINT=https://hf-mirror.com
  python -c "from local_llm import classify; classify('test')"

离线模式：
  export HF_HUB_OFFLINE=1
"""

import os
import threading
from typing import Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_model = None
_tokenizer = None
_loaded = False
_lock = threading.Lock()
_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def _ensure_loaded():
    global _model, _tokenizer, _loaded
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch

            print(f"[local_llm] 加载量化模型 {_MODEL_NAME} (4-bit)...")

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            _tokenizer = AutoTokenizer.from_pretrained(
                _MODEL_NAME, trust_remote_code=True,
            )
            _model = AutoModelForCausalLM.from_pretrained(
                _MODEL_NAME,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
            _loaded = True
            print(f"[local_llm] 加载完成（设备: {_model.device}）")
        except Exception as e:
            print(f"[local_llm] GPU 量化加载失败: {e}")
            print("[local_llm] 回退 CPU 模式...")
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                _tokenizer = AutoTokenizer.from_pretrained(
                    _MODEL_NAME, trust_remote_code=True,
                )
                _model = AutoModelForCausalLM.from_pretrained(
                    _MODEL_NAME, device_map="cpu", trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                _loaded = True
                print("[local_llm] CPU 模式加载完成")
            except Exception as e2:
                print(f"[local_llm] CPU 加载也失败: {e2}")
                print("[local_llm] 本地模型不可用，将回退到在线 API")


def _generate(prompt: str, max_new_tokens: int = 200, temperature: float = 0.1) -> str:
    if not _loaded:
        return ""
    messages = [{"role": "user", "content": prompt}]
    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    import torch
    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)
    outputs = _model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=_tokenizer.eos_token_id,
    )
    response = _tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


# ─── 公开接口 ─────────────────────────────────────────────

def classify(query: str) -> str:
    """
    任务分类。

    返回值：
      "direct"          — 不需要工具，直接回答
      "tools: xxx,yyy"  — 需要指定工具
      "general"         — 不确定，走完整 ReAct
      ""                — 模型未加载，走默认流程
    """
    if not _loaded:
        _ensure_loaded()
    if not _loaded:
        return ""

    prompt = (
        "你是一个AI助手，需要判断用户问题是否需要调用外部工具来回答。\n"
        "可用的工具有：read_file_tool, write_file_tool, shell_tool, calculator, "
        "search_files_tool, grep_tool, get_weather_tool, rag_query, web_fetch\n\n"
        "规则：\n"
        "- 如果问题纯粹是非专业领域的知识问答、定义解释、闲聊，不需要工具 → 返回 direct\n"
        "- 如果问题需要读取文件、搜索代码、查询知识库等 → 返回 tools: 工具名1,工具名2,工具名n\n"
        "- 如果不确定 → 返回 general\n\n"
        "只返回一行结果，不要多余解释。\n\n"
        f"问题：{query}"
    )
    result = _generate(prompt, max_new_tokens=100)
    result = result.strip().lower()

    if result.startswith("tools:"):
        return result
    if result in ("direct", "general"):
        return result
    return "general"


def summarize(user_msg: str, assistant_msg: str) -> str:
    """
    对话摘要。

    输入：用户问题和助手回答
    输出：一句话摘要
    """
    if not _loaded:
        _ensure_loaded()
    if not _loaded:
        return f"用户：{user_msg[:50]}… 助手：{assistant_msg[:50]}…"

    prompt = (
        "用一句话总结这段对话：用户问了什么，以及助手如何回答。\n\n"
        f"用户：{user_msg}\n"
        f"助手：{assistant_msg}\n\n"
        "总结："
    )
    return _generate(prompt, max_new_tokens=200, temperature=0.3)
