"""
agent_config.py — 全局配置中心

所有硬编码值集中于此，各模块通过 `from agent_config import config` 引用。
可通过环境变量覆盖（AGENT__xxx），无需修改代码即可调整运行参数。
"""

from dataclasses import dataclass, field
import os


def _env(key: str, default: str) -> str:
    """优先读环境变量 AGENT__{key}，否则返回 default"""
    return os.environ.get(f"AGENT__{key}", default)


# ---------------------------------------------------------------------------
# 核心连接
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """大语言模型（DeepSeek）"""
    api_key: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", ""))
    base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://api.deepseek.com"))
    model: str = "deepseek-chat"
    temperature: float = 0.3
    max_turns: int = int(_env("MAX_TURNS", "15"))


@dataclass
class LocalLLMConfig:
    """本地小模型（Qwen2.5-1.5B-Instruct）"""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    classify_temperature: float = 0.1
    classify_max_tokens: int = 100
    summarize_temperature: float = 0.3
    summarize_max_tokens: int = 200


# ---------------------------------------------------------------------------
# 嵌入 & 向量库
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingConfig:
    """嵌入模型"""
    model_name: str = "BAAI/bge-small-zh-v1.5"

    @property
    def cache_path(self) -> str:
        name = self.model_name.replace("/", "--")
        return os.path.join(
            os.path.expanduser("~/.cache/huggingface/hub"),
            f"models--{name}"
        )


@dataclass
class ChromaConfig:
    """ChromaDB 向量库"""
    db_path: str = field(default_factory=lambda: _env("DB_PATH", "./rag_data"))
    collection_name: str = "knowledge"
    long_term_collection: str = "long_term_memory"
    batch_size: int = 5000


# ---------------------------------------------------------------------------
# RAG 检索
# ---------------------------------------------------------------------------

@dataclass
class RAGConfig:
    """RAG 检索参数"""
    vector_top_k: int = 10
    hybrid_top_k: int = 5
    rrf_constant: int = 60
    distance_threshold: float = 1.0
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_max_length: int = 512
    rerank_min_docs: int = 10         # 超过此文档数才加载重排序器


# ---------------------------------------------------------------------------
# 记忆系统
# ---------------------------------------------------------------------------

@dataclass
class MemoryConfig:
    """记忆系统"""
    short_term_path: str = "agent_sessions.json"
    mid_term_path: str = "mid_term_memory.json"
    keyword_path: str = "keyword_memory.json"
    short_term_max_rounds: int = 8
    mid_term_max_items: int = 9
    mid_term_merge_batch: int = 8


# ---------------------------------------------------------------------------
# 缓存 & 工具
# ---------------------------------------------------------------------------

@dataclass
class CacheConfig:
    """语义缓存"""
    threshold: float = 0.85
    ttl_seconds: int = 300


@dataclass
class ToolConfig:
    """工具执行限制"""
    shell_timeout: int = 30
    web_fetch_timeout: int = 15
    grep_max_results: int = 200


# ---------------------------------------------------------------------------
# Web 服务
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    """FastAPI 服务"""
    host: str = field(default_factory=lambda: _env("SERVER_HOST", "0.0.0.0"))
    port: int = int(_env("SERVER_PORT", "8000"))


# ---------------------------------------------------------------------------
# 计费（DeepSeek 计价参考）
# ---------------------------------------------------------------------------

@dataclass
class BillingConfig:
    """Token 计价（元/百万 tokens）"""
    cache_hit_rate: float = 0.5   # 缓存命中
    cache_miss_rate: float = 2.0  # 缓存未命中
    output_rate: float = 8.0      # 输出


# ===========================================================================
# 全局单例
# ===========================================================================

@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    local_llm: LocalLLMConfig = field(default_factory=LocalLLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    tool: ToolConfig = field(default_factory=ToolConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    billing: BillingConfig = field(default_factory=BillingConfig)


config = Config()
