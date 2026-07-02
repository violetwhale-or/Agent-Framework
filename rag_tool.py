"""
rag_tool.py — RAG 检索工具（懒加载版）

只在第一次调用 rag_query() 时加载模型和知识库。
启动阶段不影响 agent 初始化速度。
"""

import chromadb
from sentence_transformers import SentenceTransformer

# ── 全局变量：第一次调用时初始化 ──
_model = None
_collection = None


def _ensure_initialized():
    """第一次调用时加载模型和知识库"""
    global _model, _collection
    if _model is not None:
        return

    print("🔄 [RAG] 首次加载嵌入模型...")
    _model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    docs = [
        "ReAct 循环是 Thought → Action → Observation 的交替过程",
        "ChromaDB 是一个轻量级向量数据库，无需独立服务器即可运行",
        "Python 是一种解释型、面向对象的编程语言",
    ]
    doc_vectors = _model.encode(docs)
    client_db = chromadb.PersistentClient(path="./rag_data")
    _collection = client_db.get_or_create_collection("knowledge")
    _collection.add(
        ids=["0", "1", "2"],
        embeddings=doc_vectors.tolist(),
        documents=docs,
    )
    print("✅ [RAG] 知识库初始化完成")


def rag_query(query: str) -> str:
    """从知识库检索相关段落，返回原始文本（不调 LLM，让主 Agent 自行判断）"""
    _ensure_initialized()

    # ① 嵌入问题
    q_vec = _model.encode(query)

    # ② 检索（返回多条，方便筛选）
    results = _collection.query(
        query_embeddings=[q_vec.tolist()],   # ✅ numpy → list
        n_results=3,
        include=["documents", "distances"],
    )

    docs = results["documents"][0]
    distances = results["distances"][0]

    # ③ 阈值筛选：只保留距离 ≤ 0.7 的
    useful = []
    for i in range(len(docs)):
        if distances[i] <= 0.5:
            useful.append(docs[i])
        elif distances[i] <= 0.7:
            useful.append(docs[i] + "（相似度较弱）")
        # > 0.7 丢弃

    if not useful:
        return "（知识库中未找到相关信息）"

    return "\n---\n".join(useful)


# ── 测试（只有直接运行此文件时才执行）──
if __name__ == "__main__":
    result = rag_query("什么是 ReAct 循环？")
    print(f"RAG 检索结果:\n{result}")
