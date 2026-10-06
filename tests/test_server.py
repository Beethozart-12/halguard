"""代理服务端到端测试（用假后端，不依赖真实 LLM/网络）。

验证：检索接地 -> 注入上下文 -> 调用后端 -> 生成后验证 -> flag/none 动作。
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pytest
from fastapi.testclient import TestClient

import halguard.server as S
from halguard.config import HalGuardConfig
from halguard.retrieval import RetrievalIndex


def _build_retrieval():
    retr = RetrievalIndex.new(embedder_kind="tfidf")
    retr.embedder.fit(["巴黎是法国的首都，位于塞纳河畔。"])
    embs = retr.embedder.embed(["巴黎是法国的首都，位于塞纳河畔。"])
    retr.store.add(["巴黎是法国的首都，位于塞纳河畔。"], embs)
    return retr


@pytest.fixture(autouse=True)
def patch_backend(monkeypatch):
    async def fake(client, cfg, payload):
        # 返回一个与上下文无关的"幻觉"回答
        return {
            "choices": [{"message": {"role": "assistant",
                                     "content": "苹果是一种水果，富含维生素C，原产自南美洲。"}}],
            "model": cfg.backend_model,
        }
    monkeypatch.setattr(S, "_call_backend", fake)


@pytest.fixture
def client():
    cfg = HalGuardConfig(
        backend_model="test-model",
        support_threshold=0.35,
        risk_threshold=0.30,
        action_on_risk="flag",
        append_warning=True,
        claim_extraction="heuristic",
    )
    return TestClient(S.build_app(cfg, _build_retrieval()))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["kb_chunks"] == 1


def test_flag_action_appends_warning(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "巴黎是哪国首都？"}]},
    )
    assert r.status_code == 200
    data = r.json()
    assert "halguard" in data
    hg = data["halguard"]
    assert hg["risk"] > 0.0
    assert hg["action"] == "flag"
    # flag 动作应把警告块追加到回复
    assert "HalGuard" in data["choices"][0]["message"]["content"]


def test_none_action_no_warning():
    cfg = HalGuardConfig(
        action_on_risk="none", append_warning=True, risk_threshold=0.30,
        support_threshold=0.35, backend_model="m", claim_extraction="heuristic",
    )
    app = S.build_app(cfg, _build_retrieval())
    tc = TestClient(app)
    r = tc.post("/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "巴黎？"}]})
    data = r.json()
    assert "halguard" in data
    assert data["halguard"]["action"] == "none"
    assert "HalGuard" not in data["choices"][0]["message"]["content"]
