"""HalGuard 快速演示（无需启动服务、无需后端 LLM）。

演示：摄入示例知识库 -> 检索 -> 对一段含幻觉的回答做验证。
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from halguard.retrieval import RetrievalIndex
from halguard import verify as V

# 1) 构建索引（用 TF-IDF 嵌入，零重依赖）
retr = RetrievalIndex.new(embedder_kind="tfidf")

# 示例知识库
kb = [
    "巴黎是法国的首都，位于塞纳河畔。",
    "埃菲尔铁塔是巴黎的著名地标，建于 1889 年。",
    "苹果公司由史蒂夫·乔布斯、史蒂夫·沃兹尼亚克和罗纳德·韦恩于 1976 年创立。",
]
retr.embedder.fit(kb)
embs = retr.embedder.embed(kb)
retr.store.add(kb, embs)

# 2) 检索
query = "巴黎是哪国的首都？"
hits, _ = retr.query(query, top_k=2)
print("检索到：")
for h in hits:
    print("  -", h)

# 3) 验证一段含幻觉的回答
answer = "巴黎是德国的首都。埃菲尔铁塔建于 1889 年。"
claims = V.extract_claims(answer)
report = V.score_claims(claims, retr.store.matrix[: len(hits)], retr.embedder, 0.35)
risk = V.compute_risk(report)
print(f"\n回答：{answer}")
print(f"断言数: {len(claims)}  整体风险: {risk:.2f}")
for r in report:
    tag = "❌ 疑似幻觉" if r["unsupported"] else "✅ 已支持"
    print(f"  [{tag}] (支持度 {r['support']:.2f}) {r['claim']}")
