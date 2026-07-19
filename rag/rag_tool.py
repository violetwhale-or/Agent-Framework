import os
import chromadb
import pickle
from sentence_transformers import SentenceTransformer
from agent_config import config


_collection = None
db_path = config.chroma.db_path
_bm25_data = None
_reranker = None


def _ensure_initialized():
    global _model, _collection, _bm25_data
    if _model is not None:
        return

    if not os.path.exists(os.path.join(db_path, "chroma.sqlite3")):
        raise FileNotFoundError(
            f"知识库不存在: {db_path}/chroma.sqlite3\n"
            f"请先运行: python build_knowledge.py <文档路径>"
        )

    print("🔄 [RAG] 首次加载嵌入模型...")
    model_name = config.embedding.model_name
    cache_path = config.embedding.cache_path
    if os.path.exists(cache_path):
        _model = SentenceTransformer(model_name, local_files_only=True)
    else:
        _model = SentenceTransformer(model_name)

    client_db = chromadb.PersistentClient(db_path)
    _collection = client_db.get_collection("knowledge")

    bm25_path = os.path.join(db_path, "bm25_index.pkl")
    if os.path.exists(bm25_path):
        with open(bm25_path, "rb") as f:
            _bm25_data = pickle.load(f)
        print(f"✅ BM25 索引已加载（{len(_bm25_data['texts'])} 条）")

    print("✅ [RAG] 知识库初始化完成")


def rag_query(query: str) -> str:
    _ensure_initialized()
    q_vec = _model.encode(query)

    # 向量检索
    vector_results = _collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=config.rag.vector_top_k,
        include=["documents", "distances", "metadatas"],
    )

    # 建立 text → source 映射（优先用 BM25，回退到向量元数据）
    source_from_text = {}
    if _bm25_data is not None:
        for idx, t in enumerate(_bm25_data["texts"]):
            src = ""
            if idx < len(_bm25_data["titles"]):
                src = _bm25_data["titles"][idx].get("source", "")
            source_from_text[t] = src
    else:
        vector_metas = vector_results["metadatas"][0]
        for idx, doc_text in enumerate(vector_results["documents"][0]):
            src = ""
            if idx < len(vector_metas) and vector_metas[idx]:
                src = vector_metas[idx].get("source", "")
            source_from_text[doc_text] = src

    # RRF 合并排名
    rrf_scores = {}
    for rank, doc_text in enumerate(vector_results["documents"][0]):
        rrf_scores[doc_text] = rrf_scores.get(doc_text, 0) + 1 / (config.rag.rrf_constant + rank)

    if _bm25_data is not None:
        tokenized_query = query.split()
        bm25_scores = _bm25_data["bm25"].get_scores(tokenized_query)
        bm25_top10 = sorted(
            enumerate(bm25_scores), key=lambda x: x[1], reverse=True
        )[:10]
        for rank, (doc_id, _) in enumerate(bm25_top10):
            doc_text = _bm25_data["texts"][doc_id]
            rrf_scores[doc_text] = rrf_scores.get(doc_text, 0) + 1 / (config.rag.rrf_constant + rank)

    # 取 Top3
    final_rank = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:config.rag.hybrid_top_k]
    useful = [text for text, _ in final_rank]
    if not useful:
        return "（未检索到相关信息）"

    # Rerank（数据量 > 1000 时启用）
    if _bm25_data is not None and len(_bm25_data["texts"]) > config.rag.rerank_min_docs:
        _load_reranker()
        if _reranker is not None:
            pairs = [(query, text) for text in useful]
            rerank_scores = _reranker.predict(pairs)
            reranked = sorted(
                zip(useful, rerank_scores),
                key=lambda x: x[1], reverse=True
            )
            useful = [text for text, _ in reranked]

    # 拼接结果
    final = []
    for text in useful:
        source = source_from_text.get(text, "")
        tag = f"({source})" if source else ""
        final.append(f"{tag} {text}".strip())

    return "\n\n---\n\n".join(final)


def _load_reranker():
    """首次调用时才加载 Rerank 模型（2.27GB）"""
    global _reranker
    if _reranker is not None:
        return
    from sentence_transformers import CrossEncoder
    import torch
    print("🔄 [Rerank] 加载重排序模型（首次加载较慢）...")
    model_name = config.rag.rerank_model
    cache_path = os.path.join(
        os.path.expanduser("~/.cache/huggingface/hub"),
        f"models--{model_name.replace('/', '--')}"
    )
    kwargs = {"max_length": config.rag.rerank_max_length, "device": "cuda" if torch.cuda.is_available() else "cpu"}
    if os.path.exists(cache_path):
        _reranker = CrossEncoder(model_name, local_files_only=True, **kwargs)
    else:
        _reranker = CrossEncoder(model_name, **kwargs)
    print(f"✅ [Rerank] 加载完成（设备: {_reranker.device}）")


if __name__ == "__main__":
    result = rag_query("Qwen_Proxy 应该怎么使用部署？")
    print(f"RAG 检索结果:\n{result}")
