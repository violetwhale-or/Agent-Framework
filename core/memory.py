import json
import os
import time
from typing import Optional
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
from agent_config import config

# 强制离线：模型已下载到本地缓存，不走网络请求
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


# 短期记忆

class ShortTermMemory:
    """滑动窗口短期记忆，固定保留最近 N 轮完整对话"""

    MAX_ROUNDS = config.memory.short_term_max_rounds

    def __init__(self, path: str = config.memory.short_term_path):
        self.path = path
        self._sessions: dict[str, list[dict]] = {}
        self._load()

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                self._sessions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
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

    def pop_oldest_round(self, session_id: str) -> Optional[tuple[str, str]]:
        turns = self._sessions.get(session_id, [])
        rounds = sum(1 for t in turns if t["role"] == "assistant" and not t.get("tool_calls"))
        if rounds <= self.MAX_ROUNDS:
            return None
        user_msg = None
        for i, t in enumerate(turns):
            if t["role"] == "user":
                user_msg = (i, t["content"])
            elif t["role"] == "assistant" and not t.get("tool_calls") and user_msg is not None:
                result = (user_msg[1], t["content"])
                del turns[user_msg[0]:i+1]
                self._save()
                return result
        return None


# 中期记忆

class MidTermMemory:
    """被淘汰轮次→单句摘要→合并摘要，常驻上下文（按 session 隔离）"""

    MAX_ITEMS = config.memory.mid_term_max_items

    def __init__(self, client: OpenAI, path: str = config.memory.mid_term_path):
        self.client = client
        self.path = path
        self._data: dict[str, list[str]] = {}
        self._cached_blocks: dict[str, str] = {}
        self._load()

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                self._data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._data = {}
        for sid in self._data:
            self._rebuild_cache(sid)

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def archive_round(self, session_id: str, user_msg: str, assistant_msg: str):
        """淘汰一轮时调用：本地模型生成单句摘要"""
        try:
            from core.local_llm import summarize
            summary = summarize(user_msg, assistant_msg)
        except Exception:
            summary = f"用户：{user_msg[:50]}… 助手：{assistant_msg[:50]}…"
        self._data.setdefault(session_id, []).append(summary)
        self._save()

        if len(self._data[session_id]) >= self.MAX_ITEMS:
            self._merge(session_id)

        self._rebuild_cache(session_id)

    def store_summary(self, session_id: str, summary: str):
        self._data.setdefault(session_id, []).append(summary)
        self._save()

        if len(self._data[session_id]) >= self.MAX_ITEMS:
            self._merge(session_id)

        self._rebuild_cache(session_id)

    def get_summary_for_eviction(self, session_id: str, index: int = 0) -> Optional[str]:
        summaries = self._data.get(session_id, [])
        if index < len(summaries):
            return summaries[index]
        return None

    def _merge(self, session_id: str):
        summaries = self._data[session_id]
        to_merge = summaries[:config.memory.mid_term_merge_batch]
        self._data[session_id] = summaries[config.memory.mid_term_merge_batch:] + ["、".join(to_merge)]
        self._save()

    def _rebuild_cache(self, session_id: str):
        summaries = self._data.get(session_id, [])
        prefix = "【历史对话摘要】\n" if summaries else ""
        self._cached_blocks[session_id] = prefix + "\n".join(f"- {s}" for s in summaries)

    def get_block(self, session_id: str) -> str:
        return self._cached_blocks.get(session_id, "")


# 长期记忆

class LongTermMemory:
    """淘汰轮全文归档至 ChromaDB，按需语义召回"""

    def __init__(self, db_path: str = "./rag_data", collection_name: str = "long_term_memory",
                 model: Optional[SentenceTransformer] = None):
        self._model = model
        self._model_provider = None
        self._client = chromadb.PersistentClient(path=db_path)
        try:
            self._collection = self._client.get_collection(collection_name)
        except Exception:
            self._collection = self._client.create_collection(collection_name)

    def set_model_provider(self, provider):
        self._model_provider = provider
        m = provider()
        if m is not None:
            self._model = m

    def _ensure_model(self):
        if self._model is None:
            try:
                self._model = SentenceTransformer(config.embedding.model_name, local_files_only=True)
            except Exception as e:
                print(f"[memory] 模型加载失败，长期记忆归档跳过: {e}")
                raise

    def archive(self, user_msg: str, assistant_msg: str, metadata: Optional[dict] = None):
        if self._model is None:
            if self._model_provider:
                self._model = self._model_provider()
            if self._model is None:
                try:
                    self._model = SentenceTransformer(config.embedding.model_name, local_files_only=True)
                except Exception as e:
                    print(f"[memory] 模型加载失败: {e}")
                    return

        try:
            full_text = f"用户：{user_msg}\n助手：{assistant_msg}"
            vec = self._model.encode(full_text)
            self._collection.add(
                ids=[f"mem-{time.time_ns()}"],
                embeddings=[vec.tolist()],
                documents=[full_text],
                metadatas=[metadata or {"time": time.time()}],
            )
        except Exception as e:
            print(f"[memory] 长期记忆归档失败: {e}")

    def recall(self, query: str, n: int = 3) -> list[str]:
        self._ensure_model()
        vec = self._model.encode(query)
        results = self._collection.query(
            query_embeddings=[vec.tolist()], n_results=n,
            include=["documents"],
        )
        return results["documents"][0] if results["documents"] else []

    def count(self) -> int:
        return self._collection.count()


# 关键词记忆

class KeywordMemory:
    """用户固定偏好、业务实体、全局输出规则，低频更新"""

    def __init__(self, path: str = "keyword_memory.json"):
        self.path = path
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                self._data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._data = {}

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def set(self, session_id: str, keywords: str):
        self._data[session_id] = {"keywords": keywords, "updated": time.time()}
        self._save()

    def append_keywords(self, session_id: str, new_keywords: str):
        existing = self.get(session_id)
        combined = existing + (", " if existing else "") + new_keywords
        terms = [t.strip() for t in combined.split(",") if t.strip()]
        seen = set()
        unique = []
        for t in terms:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique.append(t)
        self._data[session_id] = {"keywords": ", ".join(unique), "updated": time.time()}
        self._save()

    def get(self, session_id: str) -> str:
        entry = self._data.get(session_id)
        return entry["keywords"] if entry else ""


# 统一入口

class MemoryManager:
    """整合三级记忆 + 关键词，提供 run_stream 需要的全部接口"""

    def __init__(self, llm_client: OpenAI, short_term_path: str = "agent_sessions.json",
                 long_term_db: str = "./rag_data",
                 mid_term_path: str = "mid_term_memory.json"):
        self.short = ShortTermMemory(short_term_path)
        self.mid = MidTermMemory(llm_client, mid_term_path)
        self.long = LongTermMemory(long_term_db)
        self.keyword = KeywordMemory()
        self._client = llm_client

    def build_messages(self, session_id: Optional[str], user_input: str,
                       system_prompt: str) -> list[dict]:
        blocks = [{"role": "system", "content": system_prompt}]

        if session_id:
            mid_block = self.mid.get_block(session_id)
            if mid_block:
                blocks.append({"role": "user", "content": mid_block})
            history = self.short.load(session_id)
            blocks.extend(history)

        blocks.append({"role": "user", "content": user_input})
        return blocks

    def on_turn_complete(self, session_id: str, user_msg: str, assistant_msg: str):
        """每轮结束后调用：淘汰最旧一轮 → 中期摘要 + 长期归档"""
        evicted = self.short.pop_oldest_round(session_id)
        if evicted:
            u, a = evicted
            self.mid.archive_round(session_id, u, a)
            self.long.archive(u, a, {"session": session_id, "time": time.time()})
