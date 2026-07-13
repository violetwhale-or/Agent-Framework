"""
查看向量知识库内容

用法：python inspect_kb.py

输出：
  - 知识库总条数
  - 各来源文件的条目数
  - 每条记录的标题和字数
"""

from sentence_transformers import SentenceTransformer
import chromadb

db_path = "./rag_data"
client = chromadb.PersistentClient(path=db_path)
col = client.get_collection("knowledge")

all_data = col.get(limit=col.count(), include=["documents", "metadatas"])

print(f"\n知识库总条数: {col.count()}\n")

# 按来源统计
from collections import Counter
sources = Counter()
for m in all_data["metadatas"]:
    src = m.get("source", "未知") if m else "未知"
    sources[src] += 1

print("各来源文件条目数:")
for src, cnt in sources.most_common():
    print(f"  {src}: {cnt} 条")
print()

# 列出前 20 条
print("条目列表（前 20 条）:")
print(f"{'#':>3} {'来源':<30} {'标题':<40} {'字数'}")
print("-" * 90)
for i, (doc, meta) in enumerate(zip(all_data["documents"], all_data["metadatas"])):
    if i >= 20:
        print(f"  ... 还有 {col.count() - 20} 条")
        break
    src = meta.get("source", "?") if meta else "?"
    title = meta.get("title", "?") if meta else "?"
    print(f"{i:>3} {str(src)[:30]:<30} {str(title)[:40]:<40} {len(doc)}")
