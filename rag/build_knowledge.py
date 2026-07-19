import sys
import re
import os
import uuid
import chromadb
import pickle
import glob
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from agent_config import config


def split_by_heading(text: str, pattern: str) -> list[dict]:
    """
    按标题层级切分文本。
    pattern: r"^##\s+" / r"^###\s+" / r"^####\s+"
    返回 [{"title": "标题名", "content": "段落正文"}, ...]
    未匹配到时返回空列表。
    """
    lines = text.split("\n")
    chunks = []
    current_title = None
    current_lines = []

    for line in lines:
        m = re.match(pattern, line)
        if m:
            if current_title is not None:
                chunks.append({"title": current_title.strip(),
                               "content": "\n".join(current_lines)})
            current_title = m.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_title is not None:
        chunks.append({"title": current_title.strip(),
                       "content": "\n".join(current_lines)})
    return chunks


def _split_preserving_blocks(text: str, min_len: int = 20) -> list[str]:
    """
    按段落切分，保护 ``` 代码块和 | 表格不被 \n\n 切开。
    """
    lines = text.split("\n")
    blocks = []
    current = []
    in_code = False

    for line in lines:
        # 代码块边界 ```（保护整体）
        if line.strip().startswith("```"):
            if in_code:
                current.append(line)
                blocks.append("\n".join(current))
                current, in_code = [], False
            else:
                if current:
                    blocks.append("\n".join(current))
                    current = []
                in_code = True
                current.append(line)
            continue

        if in_code:
            current.append(line)
            continue

        # 表格行（连续 | 开头且 | 结尾 → 统一归入表格块）
        current.append(line)

    if current:
        blocks.append("\n".join(current))

    # 展开非保护的段落
    result = []
    for blk in blocks:
        blk = blk.strip()
        if not blk:
            continue
        # 代码块或表格块 → 整块保留
        if blk.startswith("```") or all(
            l.strip().startswith("|") and l.strip().endswith("|")
            for l in blk.split("\n") if l.strip()
        ):
            if len(blk) > min_len:
                result.append(blk)
        else:
            # 普通段落按 \n\n 切
            for p in blk.split("\n\n"):
                p = p.strip()
                if len(p) > min_len:
                    result.append(p)
    return result


def _recursive_chunk(content: str, title: str,
                     patterns: list[str], max_size: int) -> list[dict]:
    """
    递归切割核心：

    content  — 待切文本
    title    — 继承的父标题路径
    patterns — 剩余待尝试的标题层级，如 ["^###\\s+", "^####\\s+"]
    max_size — 单块字数上限

    逻辑：
      ① 字数 ≤ max_size → 直接保留
      ② 无更多标题层级 → 按 \n\n 段落切分（兜底）
      ③ 尝试当前层级 → 没切动就降级 → 切动了就逐块递归
    """
    # ① 够小，直接保留
    if len(content) <= max_size:
        return [{"title": title, "content": content}]

    # ② 标题层级耗尽，按段落切分（保护代码块和表格）
    if not patterns:
        pieces = _split_preserving_blocks(content, min_len=20)
        if not pieces:
            return [{"title": title, "content": content[:max_size]}]
        return [{"title": title, "content": p} for p in pieces]

    # ③ 尝试当前标题层级
    subs = split_by_heading(content, patterns[0])

    # 没切动（无匹配 / 切完还是同一块）→ 降级到下一层
    if not subs or (len(subs) == 1 and subs[0]["content"].strip() == content.strip()):
        return _recursive_chunk(content, title, patterns[1:], max_size)

    # 切动了 → 对每块递归（带上更细的标题层级）
    result = []
    for sub in subs:
        sub_title = f"{title} > {sub['title']}" if title else sub["title"]
        result.extend(_recursive_chunk(sub["content"], sub_title,
                                       patterns[1:], max_size))
    return result


def chunk_markdown_recursive(filepath: str, max_size: int = 1000) -> list[dict]:
    """
    入口：读取 .md 文件，按 ## → ### → #### → \n\n 逐层切割。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # 第一层：按 ## 切出顶部分区
    top_sections = split_by_heading(text, r"^##\s+(.+)$")
    if not top_sections:
        top_sections = [{"title": os.path.basename(filepath), "content": text}]

    # 后续层级
    remaining = [r"^###\s+(.+)$", r"^####\s+(.+)$"]

    result = []
    for sec in top_sections:
        result.extend(_recursive_chunk(sec["content"], sec["title"],
                                       remaining, max_size))
    return result


def build_knowledge(chunks: list[dict], db_path: str = config.chroma.db_path, doc_source = None):
    """编码所有段落，存入 ChromaDB（先删后建，避免追加写入不可靠）"""
    print("🔄 加载嵌入模型...")
    model_name = "BAAI/bge-small-zh-v1.5"
    cache_path = os.path.join(
        os.path.expanduser("~/.cache/huggingface/hub"),
        f"models--{model_name.replace('/', '--')}"
    )
    if os.path.exists(cache_path):
        model = SentenceTransformer(model_name, local_files_only=True)
    else:
        model = SentenceTransformer(model_name)
    texts = [f"{c['title']}\n{c['content']}" for c in chunks]
    meta = [{"title": c["title"], "source": doc_source or c.get("source", "")} for c in chunks]

    print(f"🔢 编码 {len(texts)} 个段落...")
    vectors = model.encode(texts)

    client = chromadb.PersistentClient(path=db_path)
    # 先删后建：确保重建时不会累积旧数据
    try:
        client.delete_collection("knowledge")
        print("已删除旧知识库，重新构建...")
    except Exception:
        pass
    col = client.create_collection("knowledge")

    # 分批写入（ChromaDB 单次上限约 5461 条）
    BATCH_SIZE = 5000
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(chunks))
        col.add(
            ids=[str(uuid.uuid4()) for _ in range(batch_end - i)],
            embeddings=vectors.tolist()[i:batch_end],
            documents=texts[i:batch_end],
            metadatas=meta[i:batch_end],
        )
        print(f"  ... 写入 {batch_end}/{len(chunks)} 条")
    print(f"✅ 构建完成：{len(chunks)} 条，存储在 {db_path}/")

    # 中文分词（jieba）后计算 BM25 索引
    tokenized_corpus = [list(jieba.cut(c["content"])) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    index_path = os.path.join(db_path, "bm25_index.pkl")
    with open(index_path, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "texts": texts,
            "titles": meta,
        }, f)
    print(f"✅ BM25 索引已保存（{len(chunks)} 条）→ {index_path}")


def build_from_directory(dir_path: str, db_path: str = "./rag_data", max_size: int = 1000):
    """读取目录下所有 .md 文件，逐个切片后一并构建知识库"""
    md_files = sorted(glob.glob(os.path.join(dir_path, "*.md")))
    if not md_files:
        print(f"目录下没有 .md 文件: {dir_path}")
        return

    all_chunks = []
    for fp in md_files:
        fname = os.path.basename(fp)
        print(f"📖 {fname}")
        chunks = chunk_markdown_recursive(fp, max_size)
        for c in chunks:
            c["source"] = fname
        all_chunks.extend(chunks)
        print(f"   → {len(chunks)} 块")

    build_knowledge(all_chunks, db_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python build_knowledge.py <文件或目录路径>")
        print("      python build_knowledge.py ./知识库目录/")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"❌ 路径不存在: {path}")
        sys.exit(1)

    if os.path.isdir(path):
        build_from_directory(path)
    else:
        print(f"📖 加载文档: {path}")
        chunks = chunk_markdown_recursive(path)
        print(f"📄 共切分为 {len(chunks)} 块：")
        build_knowledge(chunks, doc_source=os.path.basename(path))
