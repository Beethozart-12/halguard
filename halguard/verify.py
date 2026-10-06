"""生成后验证：从回答中抽取事实性断言，并对照检索上下文评估语义支持度。

核心判据：每条断言与"检索到的上下文块"的最大余弦相似度。相似度低于 support_threshold
即视为"缺乏依据"（疑似幻觉）。整体风险 = 不受支持断言数 / 断言总数。
"""
from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

QUESTION_STARTERS = ("what", "who", "when", "where", "why", "how", "which", "is", "are", "do", "does", "can", "could", "would", "will", "should")

# 明显非事实性（观点/语气）的弱信号词，命中且较短时倾向于跳过
SOFT_FILLER = ("i think", "in my opinion", "maybe", "perhaps", "it seems", "generally", "usually")


def extract_claims(text: str) -> List[str]:
    """启发式抽取事实性断言（句子级）。"""
    if not text:
        return []
    # 去掉代码块，避免把代码当事实
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    raw = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    claims: List[str] = []
    for s in raw:
        s = s.strip(" \t\n-•*→ ")
        if len(s) < 6:
            continue
        low = s.lower()
        # 跳过问句
        if low.endswith("?") or low.endswith("？"):
            continue
        if low.split(" ", 1)[0] in QUESTION_STARTERS and len(s) < 60:
            continue
        # 跳过纯观点（弱信号 + 短）
        if any(f in low for f in SOFT_FILLER) and len(s) < 50:
            continue
        # 至少要包含实体/数字/专有名词迹象：大写字母或数字
        if not re.search(r"[A-Z0-9\u4e00-\u9fff]", s):
            continue
        claims.append(s)
    return claims


def score_claims(claims: List[str], chunk_embeddings: np.ndarray, embedder, threshold: float) -> List[Dict]:
    """为每条断言计算最大语义支持度，并判定是否不受支持。"""
    results: List[Dict] = []
    if not claims:
        return results
    ce = np.asarray(embedder.embed(claims), dtype=np.float32)
    norms = np.linalg.norm(ce, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    ce = ce / norms

    if chunk_embeddings is None or len(chunk_embeddings) == 0:
        # 无上下文可对照 -> 全部标记为不受支持（无法佐证）
        for c in claims:
            results.append({"claim": c, "support": 0.0, "unsupported": True})
        return results

    M = np.asarray(chunk_embeddings, dtype=np.float32)
    mn = np.linalg.norm(M, axis=1, keepdims=True)
    mn[mn == 0] = 1.0
    M = M / mn

    sims = M @ ce.T  # (n_chunks, n_claims)
    best = np.max(sims, axis=0)  # (n_claims,)
    for c, b in zip(claims, best):
        results.append({"claim": c, "support": round(float(b), 4), "unsupported": bool(b < threshold)})
    return results


def compute_risk(report: List[Dict]) -> float:
    if not report:
        return 0.0
    unsupported = sum(1 for r in report if r["unsupported"])
    return round(unsupported / len(report), 4)


def redact(content: str, report: List[Dict]) -> str:
    """将不受支持的断言替换为占位标记。"""
    out = content
    for r in report:
        if r["unsupported"] and r["claim"] in out:
            out = out.replace(r["claim"], "[已移除：缺乏依据]")
    return out


def build_warning(report: List[Dict], risk: float) -> str:
    lines = ["⚠️ HalGuard 提示：本次回答中存在以下缺乏上下文依据、疑似幻觉的断言："]
    for r in report:
        if r["unsupported"]:
            lines.append(f"  - (支持度 {r['support']:.2f}) {r['claim']}")
    lines.append(f"整体幻觉风险评分：{risk:.2f}")
    return "\n".join(lines)
