"""联网搜索开关的服务器端到端测试（搜索 HTTP 已 mock，仅后端链路真实）。"""
import os
import sys
import time
import threading

import pytest

_HERE = os.path.dirname(__file__)
MODEL = os.path.abspath(os.path.join(_HERE, "..", "..", "halguard_models", "paraphrase-multilingual-MiniLM-L12-v2"))
HAS_MODEL = os.path.isdir(MODEL)

pytestmark = pytest.mark.skipif(
    not HAS_MODEL,
    reason="本地多语种模型未找到，跳过联网搜索开关端到端测试",
)

import httpx
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import uvicorn

import halguard.server as S
from halguard.config import HalGuardConfig
from halguard.retrieval import RetrievalIndex
from halguard.server import build_app

PORT = 18925
FAKE_RESULTS = [
    {"title": "埃菲尔铁塔 - 维基百科", "url": "https://example.com/eiffel", "snippet": "埃菲尔铁塔建成于1889年，高约330米。"},
    {"title": "Eiffel Tower Facts", "url": "https://example.org/tower", "snippet": "Completed in 1889, 330 meters tall."},
]


@pytest.fixture(scope="module")
def retrieval():
    return RetrievalIndex.new(embedder_kind="st", st_model=MODEL)


@pytest.fixture(scope="module")
def backend_captured():
    """带请求捕获的 mock 后端。"""
    captured = []
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def cc(req: Request):
        captured.append(await req.json())
        return {"choices": [{"message": {"role": "assistant", "content": "根据搜索结果，埃菲尔铁塔建成于1889年[1]。"}}], "model": "mock"}

    cfg = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            httpx.get(f"http://127.0.0.1:{PORT}/", timeout=1.0)
            break
        except Exception:
            time.sleep(0.3)
    yield captured
    server.should_exit = True


def _make_client(retrieval, backend_captured, **cfg_kw):
    cfg = HalGuardConfig(
        backend_base_url=f"http://127.0.0.1:{PORT}/v1",
        backend_api_key="mock",
        backend_model="mock",
        embedder="st",
        st_model=MODEL,
        action_on_risk="none",
        claim_extraction="heuristic",
        **cfg_kw,
    )
    return TestClient(build_app(cfg, retrieval))


def test_websearch_on_injects_search_results(retrieval, backend_captured, monkeypatch):
    monkeypatch.setattr(S, "_web_search", lambda q, **kw: FAKE_RESULTS)
    backend_captured.clear()
    client = _make_client(retrieval, backend_captured)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "mock", "messages": [{"role": "user", "content": "埃菲尔铁塔多高？"}], "websearch": True},
    )
    assert r.status_code == 200, r.text
    data = r.json()

    # 1) 后端收到了强制联网系统提示 + 搜索结果
    assert backend_captured, "后端未收到请求"
    sys_msgs = [m for m in backend_captured[0]["messages"] if m.get("role") == "system"]
    assert sys_msgs, "未注入 system 消息"
    assert "网络搜索结果" in sys_msgs[0]["content"], sys_msgs[0]["content"][:200]
    assert "埃菲尔铁塔建成于1889年" in sys_msgs[0]["content"]

    # 2) websearch 开关字段未透传给后端
    assert "websearch" not in backend_captured[0], "开关字段不应透传给后端"

    # 3) 元数据包含来源
    ws = data["halguard"]["websearch"]
    assert ws["enabled"] is True and ws["failed"] is False
    assert ws["sources"][0]["url"] == "https://example.com/eiffel"


def test_websearch_failure_shows_honesty_block(retrieval, backend_captured, monkeypatch):
    def boom(q, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(S, "_web_search", boom)
    backend_captured.clear()
    client = _make_client(retrieval, backend_captured)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "mock", "messages": [{"role": "user", "content": "今天有什么新闻？"}], "websearch": True},
    )
    assert r.status_code == 200, r.text
    sys_msgs = [m for m in backend_captured[0]["messages"] if m.get("role") == "system"]
    assert sys_msgs and "网络搜索失败" in sys_msgs[0]["content"], sys_msgs[0]["content"][:200]
    ws = r.json()["halguard"]["websearch"]
    assert ws["enabled"] is True and ws["failed"] is True and ws["sources"] == []


def test_websearch_off_uses_kb_context(retrieval, backend_captured, monkeypatch):
    """开关关闭（默认）：不联网，走普通 RAG 注入；请求级字段可显式关闭。"""
    monkeypatch.setattr(S, "_web_search", lambda q, **kw: FAKE_RESULTS)  # 若误调用会污染断言
    backend_captured.clear()
    client = _make_client(retrieval, backend_captured, websearch_enabled=False)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "mock", "messages": [{"role": "user", "content": "巴黎的首都是哪里？"}], "websearch": False},
    )
    assert r.status_code == 200, r.text
    assert backend_captured, "后端未收到请求"
    sys_msgs = [m for m in backend_captured[0]["messages"] if m.get("role") == "system"]
    # 无知识库内容（空索引）且未联网 -> 无 system 接地块
    ws = r.json()["halguard"]["websearch"]
    assert ws["enabled"] is False and ws["sources"] == []


def test_websearch_config_env(monkeypatch):
    """HALGUARD_WEBSEARCH=1 开启全局默认。"""
    monkeypatch.setenv("HALGUARD_WEBSEARCH", "1")
    monkeypatch.setenv("HALGUARD_WEBSEARCH_MAX_RESULTS", "8")
    cfg = HalGuardConfig.load()
    assert cfg.websearch_enabled is True
    assert cfg.websearch_max_results == 8
    monkeypatch.delenv("HALGUARD_WEBSEARCH")
    monkeypatch.delenv("HALGUARD_WEBSEARCH_MAX_RESULTS")
    assert HalGuardConfig.load().websearch_enabled is False


def test_websearch_empty_content_retries_with_bigger_budget(retrieval, backend_captured, monkeypatch):
    """思考模型预算耗尽（content 空、reasoning 非空）时，服务端应双倍预算重试并取回正文。"""
    monkeypatch.setattr(S, "_web_search", lambda q, **kw: FAKE_RESULTS)
    calls = []

    async def fake_backend(client, cfg, payload):
        calls.append(dict(payload))
        if len(calls) == 1:
            return {"choices": [{"message": {"role": "assistant", "content": "", "reasoning": "我想想……"}}]}
        return {"choices": [{"message": {"role": "assistant", "content": "根据搜索结果，埃菲尔铁塔高约330米[1]。"}}]}

    monkeypatch.setattr(S, "_call_backend", fake_backend)
    client = _make_client(retrieval, backend_captured, websearch_min_max_tokens=4096)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "mock", "messages": [{"role": "user", "content": "埃菲尔铁塔多高？"}],
              "max_tokens": 500, "websearch": True},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["choices"][0]["message"]["content"].strip(), "重试后应拿到正文"
    # 1) 首次调用即被抬到预算下限（请求里只给了 500）
    assert calls[0]["max_tokens"] == 4096
    # 2) 重试调用预算翻倍
    assert len(calls) == 2 and calls[1]["max_tokens"] == 4096 * 2


def test_websearch_budget_floor_applied(retrieval, backend_captured, monkeypatch):
    """联网模式下 max_tokens 低于下限会被抬高；关闭开关则不动。"""
    monkeypatch.setattr(S, "_web_search", lambda q, **kw: FAKE_RESULTS)
    calls = []

    async def fake_backend(client, cfg, payload):
        calls.append(dict(payload))
        return {"choices": [{"message": {"role": "assistant", "content": "好的。"}}]}

    monkeypatch.setattr(S, "_call_backend", fake_backend)
    client = _make_client(retrieval, backend_captured, websearch_min_max_tokens=4096)
    # 开关开：预算被抬高
    client.post("/v1/chat/completions",
                json={"model": "mock", "messages": [{"role": "user", "content": "问题"}],
                      "max_tokens": 128, "websearch": True})
    assert calls[-1]["max_tokens"] == 4096
    # 开关关：尊重调用方给的 max_tokens
    client.post("/v1/chat/completions",
                json={"model": "mock", "messages": [{"role": "user", "content": "问题"}],
                      "max_tokens": 128, "websearch": False})
    assert calls[-1]["max_tokens"] == 128


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
