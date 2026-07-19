"""
eval_rag.py — RAG 检索评估

用法：python eval_rag.py

评估内容：
  - 纯语义检索 vs 混合检索（语义 + BM25 + RRF）
  - Recall@1 / Recall@3
  - 平均返回结果数（阈值过滤效果）
"""

import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.rag_tool import _ensure_initialized
from rag import rag_tool

# ─── 测试集 ──────────────────────────────────────────────
# 80 题，覆盖 RAG_initial_markdown/ 下全部 6 篇中文文档
test_set = [
    # ======== Qwen-Proxy (10) ========
    ("Qwen-Proxy 是什么", ["Qwen", "代理"]),
    ("怎么用 Docker 部署 Qwen-Proxy", ["Docker", "部署"]),
    ("如何配置 API_KEY", ["API_KEY"]),
    ("环境变量 SERVICE_PORT 的作用", ["SERVICE_PORT", "端口"]),
    ("多 API_KEY 怎么配置", ["API_KEY", "多个"]),
    ("缓存模式有哪几种", ["CACHE_MODE", "缓存"]),
    ("流式输出怎么开启", ["流式", "stream"]),
    ("怎么看 Qwen-Proxy 的日志", ["LOG_LEVEL", "日志"]),
    ("Anthropic 格式怎么调用", ["/v1/messages", "Anthropic"]),
    ("Qwen-Proxy 支持哪些模型", ["qwen", "model"]),

    # ======== solve_problem (12) ========
    ("什么是 ReAct 框架", ["ReAct", "思考"]),
    ("RAG 和 Agent 的区别是什么", ["RAG", "Agent", "区别"]),
    ("怎么降低大模型幻觉", ["幻觉", "RAG"]),
    ("上下文窗口超限怎么处理", ["上下文窗口", "Token"]),
    ("短期记忆和长期记忆的区别", ["短期", "长期", "记忆"]),
    ("Agent 工具调用失败怎么优化", ["工具", "失败"]),
    ("什么是 CoT 思维链", ["CoT", "思维链"]),
    ("微调与 RAG 的适用场景", ["微调", "RAG"]),
    ("大模型幻觉的根本成因有哪些", ["幻觉", "概率生成"]),
    ("Agent 面试有几轮", ["一面", "二面", "三面"]),
    ("多工具同时调用冲突怎么解决", ["工具冲突", "串行"]),
    ("向量数据库在 RAG 中承担什么角色", ["向量数据库", "检索"]),

    # ======== agent_develop (16) ========
    ("AI Agent 的范式转移是什么", ["范式转移", "Agent"]),
    ("2025-2026 年 Agent 爆发的关键转折点", ["Function Calling", "2023"]),
    ("Agent 三大核心能力是什么", ["感知", "决策", "执行"]),
    ("什么是 ReAct 循环", ["ReAct", "观察", "行动"]),
    ("Agent 的短期记忆如何实现", ["短期记忆", "上下文窗口"]),
    ("Agent 的长期记忆如何实现", ["长期记忆", "向量数据库"]),
    ("什么是冷热分层记忆", ["冷热分层", "短期", "长期"]),
    ("工具调用的准确率怎么优化", ["工具描述", "参数校验"]),
    ("多工具冲突怎么设计规避逻辑", ["串行", "前置校验"]),
    ("复杂任务自主规划怎么实现", ["CoT", "任务拆解"]),
    ("什么是 Zero-shot CoT", ["Zero-shot", "一步步思考"]),
    ("ReAct 和 CoT 的区别", ["ReAct", "CoT", "工具"]),
    ("为什么 Agent 需要反思校验机制", ["反思", "校验"]),
    ("Agent 在企业的实际落地场景", ["客服", "自动化", "流程"]),
    ("Agent 的成本控制有哪些策略", ["Token", "缓存", "预算"]),
    ("Agent 的安全风险怎么防范", ["安全", "权限", "注入"]),

    # ======== industrial (14) ========
    ("工业软件的核心交互范式是什么", ["WIMP", "菜单", "工具栏"]),
    ("传统工业软件的断层在哪里", ["定制化", "僵化", "柔性制造"]),
    ("AI 如何改变工业软件", ["AI", "自然语言", "自动化"]),
    ("什么是 CAD 和 CAM", ["CAD", "CAM", "工程"]),
    ("EDA 工程师的痛点是什么", ["EDA", "布线", "网络节点"]),
    ("什么是 MES 系统", ["MES", "制造执行"]),
    ("PLC 编程的挑战是什么", ["PLC", "梯形图"]),
    ("工业大模型和通用大模型的区别", ["工业", "通用", "领域"]),
    ("数字孪生在工业中的应用", ["数字孪生", "仿真"]),
    ("工业 AI 的安全要求", ["安全", "实时", "稳定"]),
    ("工业软件的 AI 落地路径", ["AutoCAD", "AI", "集成"]),
    ("柔性制造需要什么", ["快速切换", "小批量", "多品种"]),
    ("AI Agent 在工业场景中的角色", ["Agent", "工业", "自动化"]),
    ("工业数据的隐私和保护", ["数据", "隐私", "本地"]),

    # ======== Harness (14) ========
    ("Harness Engineering 是什么", ["Harness", "软件交付", "平台"]),
    ("Harness 的核心公式是什么", ["控制", "约束", "驱动"]),
    ("什么是 Policy-as-Code", ["策略即代码", "Policy"]),
    ("金丝雀发布是什么", ["金丝雀", "发布"]),
    ("蓝绿部署的原理是什么", ["蓝绿", "部署"]),
    ("滚动更新和蓝绿部署的区别", ["滚动更新", "蓝绿"]),
    ("自动化门禁在部署中的作用", ["门禁", "自动化", "验证"]),
    ("什么是 Chaos Engineering", ["混沌工程", "Chaos"]),
    ("可观测性的三大支柱", ["日志", "指标", "链路"]),
    ("Harness 中的 AI 驱动决策", ["AI", "策略", "自动化"]),
    ("部署失败怎么自动回滚", ["回滚", "自动", "失败"]),
    ("Harness 的多环境管理", ["环境", "开发", "生产"]),
    ("安全和合规在 Harness 中怎么保证", ["安全", "合规", "策略"]),
    ("Harness 和传统 CI/CD 的区别", ["Harness", "CI/CD", "平台"]),

    # ======== work_for_agent (14) ========
    ("2026 年 AI 人才需求结构的变化", ["结构性重构", "落地"]),
    ("字节跳动 2026 春招 AI 岗位分布", ["字节跳动", "7000"]),
    ("腾讯春招技术岗扩招情况", ["腾讯", "扩招", "36%"]),
    ("阿里 AI Coding 团队是什么", ["阿里", "AI Coding"]),
    ("华为 AI 实习生的覆盖方向", ["华为", "AI Infra", "大模型"]),
    ("AI 岗位从模糊到精细的趋势", ["Agent工程师", "算法工程师"]),
    ("Agent 工程师的核心职责", ["Agent", "工程", "落地"]),
    ("Agent 算法工程师的核心职责", ["Agent", "算法", "模型"]),
    ("AI 产品经理需要什么能力", ["产品经理", "技术"]),
    ("科锐国际 2026 薪酬指南要点", ["科锐国际", "薪酬"]),
    ("AI Infra 工程师负责什么", ["AI Infra", "基础设施"]),
    ("大模型架构师的定位", ["大模型架构师", "架构"]),
    ("AI 安全方向的需求趋势", ["AI安全", "安全"]),
    ("2026 AI 岗位薪酬范围参考", ["薪酬", "年薪", "范围"]),
]


def _vector_only_search(query: str, n: int = 20) -> list[str]:
    """纯向量检索（无 BM25、无 RRF）"""
    _ensure_initialized()
    q_vec = rag_tool._model.encode(query)
    results = rag_tool._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=n,
        include=["documents", "distances"],
    )
    docs = results["documents"][0]
    dists = results["distances"][0]
    filtered = [d for d, dist in zip(docs, dists) if dist <= 1.0]
    return filtered if filtered else docs[:1]


def _hybrid_search(query: str) -> list[str]:
    """混合检索：调用 rag_query 并解析返回结果"""
    from rag.rag_tool import rag_query as rq
    raw = rq(query)
    parts = [p.strip() for p in raw.split("\n\n---\n\n") if p.strip()]
    return parts


def _hit(results: list[str], keywords: list[str]) -> bool:
    """判断检索结果是否命中任一关键词"""
    combined = " ".join(results).lower()
    for kw in keywords:
        if kw.lower() in combined:
            return True
    return False


def evaluate():
    _ensure_initialized()
    total = len(test_set)
    col = rag_tool._collection

    bm25_info = f"BM25: ✓ ({len(rag_tool._bm25_data['texts'])} 条)" if rag_tool._bm25_data else "BM25: ✗"

    print(f"\n{'='*60}")
    print(f"  RAG 检索评估 — {total} 个问题")
    print(f"  来源: Qwen-Proxy(10) + solve_problem(12) + agent_develop(16)")
    print(f"        + industrial(14) + Harness(14) + work_for_agent(14)")
    print(f"  知识库: {col.count()} 条记录")
    print(f"  {bm25_info}")
    print(f"{'='*60}\n")

    # ─── 纯语义检索 ───
    print("▌ 纯语义检索（向量 + 阈值 1.0）")
    print("-" * 40)
    vec_hits_1 = 0
    vec_hits_5 = 0
    vec_total_time = 0.0

    for i, (question, keywords) in enumerate(test_set, 1):
        t0 = time.time()
        results = _vector_only_search(question, n=20)
        elapsed = time.time() - t0
        vec_total_time += elapsed

        top1 = results[:1]
        top5 = results[:5]

        hit_1 = _hit(top1, keywords)
        hit_5 = _hit(top5, keywords)

        if hit_1:
            vec_hits_1 += 1
            vec_hits_5 += 1
        elif hit_5:
            vec_hits_5 += 1

        icon = "✅" if hit_1 else ("◐" if hit_5 else "❌")
        print(f"  {icon} [{i:2d}/{total}] {question[:60]:60s}  {elapsed:.2f}s")

    print(f"\n  → Recall@1: {vec_hits_1}/{total} = {vec_hits_1/total:.0%}")
    print(f"  → Recall@5: {vec_hits_5}/{total} = {vec_hits_5/total:.0%}")
    print(f"  → 平均耗时: {vec_total_time/total:.2f}s\n")

    # ─── 混合检索 ───
    print("▌ 混合检索（语义 + BM25 + RRF + 阈值 1.0）")
    print("-" * 40)
    hyb_hits_1 = 0
    hyb_hits_5 = 0
    hyb_total_time = 0.0

    for i, (question, keywords) in enumerate(test_set, 1):
        t0 = time.time()
        results = _hybrid_search(question)
        elapsed = time.time() - t0
        hyb_total_time += elapsed

        top1 = results[:1]
        top5 = results[:5]

        hit_1 = _hit(top1, keywords)
        hit_5 = _hit(top5, keywords)

        if hit_1:
            hyb_hits_1 += 1
            hyb_hits_5 += 1
        elif hit_5:
            hyb_hits_5 += 1

        icon = "✅" if hit_1 else ("◐" if hit_5 else "❌")
        print(f"  {icon} [{i:2d}/{total}] {question[:60]:60s}  {elapsed:.2f}s  ({len(results)}段)")

    print(f"\n  → Recall@1: {hyb_hits_1}/{total} = {hyb_hits_1/total:.0%}")
    print(f"  → Recall@5: {hyb_hits_5}/{total} = {hyb_hits_5/total:.0%}")
    print(f"  → 平均耗时: {hyb_total_time/total:.2f}s\n")

    # ─── 对比汇总 ───
    print("▌ 对比汇总")
    print("=" * 40)
    print(f"{'指标':<20} {'纯语义':>12} {'混合检索':>12}")
    print("-" * 44)
    print(f"{'Recall@1':<20} {vec_hits_1/total:>11.0%} {hyb_hits_1/total:>11.0%}")
    print(f"{'Recall@5':<20} {vec_hits_5/total:>11.0%} {hyb_hits_5/total:>11.0%}")
    print(f"{'平均耗时':<20} {vec_total_time/total:>9.2f}s {hyb_total_time/total:>9.2f}s")
    print(f"\n{'='*60}")

    # Rerank 说明
    if rag_tool._bm25_data and len(rag_tool._bm25_data["texts"]) > 10:
        print("  Rerank: 已启用")
    else:
        print(f"  Rerank: 未启用（知识库 {col.count()} 条，需 > 10 条）")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    evaluate()
