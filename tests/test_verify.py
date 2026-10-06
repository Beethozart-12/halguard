from halguard.verify import extract_claims, score_claims, compute_risk, redact
from halguard.embeddings import TFIDFEmbedder


def test_extract_claims_filters_questions():
    claims = extract_claims("The sky is blue. What color is the sky?")
    assert len(claims) == 1
    assert "sky is blue" in claims[0].lower()


def test_extract_claims_splits_chinese_sentences():
    # 中文无空格，必须用 。！？ 直接切分，否则两句话会被当成一个断言漏检
    text = "巴黎是德国的首都。巴黎是法国的首都，位于塞纳河畔。"
    claims = extract_claims(text)
    assert len(claims) == 2, f"中文未按句号切分：{claims}"
    # 问句应被跳过
    claims2 = extract_claims("巴黎的首都是哪里？巴黎是法国的首都。")
    assert len(claims2) == 1
    assert "巴黎是法国的首都" in claims2[0]


def test_score_claims_supported_vs_not():
    emb = TFIDFEmbedder().fit(["Paris is the capital of France.", "The cat sat on the mat."])
    chunk_embs = emb.embed(["Paris is the capital of France."])
    rep_sup = score_claims(["Paris is the capital of France."], chunk_embs, emb, 0.35)
    assert rep_sup[0]["unsupported"] is False
    # 与上下文无任何词重叠 -> TF-IDF 支持度接近 0 -> 判定疑似幻觉
    rep_uns = score_claims(["Bananas are yellow and grow on trees."], chunk_embs, emb, 0.35)
    assert rep_uns[0]["unsupported"] is True


def test_compute_risk_empty():
    assert compute_risk([]) == 0.0


def test_redact_replaces_unsupported():
    emb = TFIDFEmbedder().fit(["Paris is the capital of France."])
    chunk_embs = emb.embed(["Paris is the capital of France."])
    content = "Bananas are yellow and grow on trees. Paris is the capital of France."
    report = score_claims(
        ["Bananas are yellow and grow on trees.", "Paris is the capital of France."],
        chunk_embs,
        emb,
        0.35,
    )
    out = redact(content, report)
    assert "Bananas" not in out
    assert "Paris is the capital of France." in out
