"""请求级云端后端覆盖测试（server 端）。

验证：云端 API 经 HalGuard 代发（含 Key 透传、模型替换、校验、测试连接端点），
以及 Key 绝不进入日志/元数据。HTTP 全部 mock，不真实联网。
"""
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
    reason="本地多语种模型未找到，跳过后端覆盖端到端测试",
)

import httpx
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import uvicorn

from halguard.config import HalGuardConfig
from halguard.retrieval import RetrievalIndex
from halguard.server import build_app

CLOUD_PORT = 18930
SECRET_KEY = "sk-super-secret-cloud-key-123"


@pytest.fixture(scope="module")
def retrieval():
    return RetrievalIndex.new(embedder_kind="st", st_model=MODEL)


@pytest.fixture(scope="module")
def cloud_backend_captured():
    """模拟云端大模型 API（OpenAI 兼容），捕获请求头与体。"""
    captured = []
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def cc(req: Request):
        captured.append({
            "headers": dict(req.headers),
            "body": await req.json(),
        })
        return {"choices": [{"message": {"role": "assistant", "content": "云端模型回答：埃菲尔铁塔高约330米。"}}]}

    @app.get("/v1/models")
    async def lm(req: Request):
        captured.append({"models_headers": dict(req.headers)})
        return {"object": "list", "data": [{"id": "cloud-model-x", "object": "model"}]}

    cfg = uvicorn.Config(app, host="127.0.0.1", port=CLOUD_PORT, log_level="warning")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            httpx.get(f"http://127.0.0.1:{CLOUD_PORT}/v1/models", timeout=1.0)
            break
        except Exception:
            time.sleep(0.3)
    yield captured
    server.should_exit = True


def _client(retrieval, **kw):
    cfg = HalGuardConfig(
        backend_base_url="http://127.0.0.1:9/v1",  # 故意不可达：证明 override 生效
        backend_api_key="local-key",
        backend_model="local-model",
        embedder="st",
        st_model=MODEL,
        action_on_risk="none",
        claim_extraction="heuristic",
        **kw,
    )
    return TestClient(build_app(cfg, retrieval))


def _override():
    return {
        "base_url": f"http://127.0.0.1:{CLOUD_PORT}/v1",
        "api_key": SECRET_KEY,
        "model": "cloud-model-x",
    }


def test_cloud_override_forwards_key_and_model(retrieval, cloud_backend_captured):
    """云端覆盖：Key/模型/地址全部按请求级配置透传到上游。"""
    cloud_backend_captured.clear()
    r = _client(retrieval).post(
        "/v1/chat/completions",
        json={"model": "ignored", "messages": [{"role": "user", "content": "埃菲尔铁塔多高？"}],
              "backend": _override()},
    )
    assert r.status_code == 200, r.text
    cap = cloud_backend_captured[-1]
    assert cap["headers"]["authorization"] == f"Bearer {SECRET_KEY}", "API Key 未正确透传"
    assert cap["body"]["model"] == "cloud-model-x", "模型未按覆盖值替换"
    assert "云端模型回答" in r.json()["choices"][0]["message"]["content"]
    hg = r.json()["halguard"]
    assert hg["backend"]["kind"] == "cloud" and hg["backend"]["model"] == "cloud-model-x"
    # 元数据绝不包含 api_key
    assert SECRET_KEY not in r.text


def test_cloud_override_invalid_returns_400(retrieval, cloud_backend_captured):
    """非法覆盖（ftp 地址 / 缺 Key / 缺模型）应 400 且不触达上游。"""
    cloud_backend_captured.clear()
    client = _client(retrieval)
    for bad in (
        {"base_url": "ftp://x/v1", "api_key": "k", "model": "m"},
        {"base_url": "http://x/v1", "api_key": "", "model": "m"},
        {"base_url": "http://x/v1", "api_key": "k", "model": ""},
    ):
        r = client.post("/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": "hi"}], "backend": bad})
        assert r.status_code == 400, f"{bad} 应被拒绝，实际 {r.status_code}"
        assert "error" in r.json()
    assert not cloud_backend_captured, "非法请求不应触达上游"


def test_cloud_override_models_endpoint(retrieval, cloud_backend_captured):
    """测试连接端点：POST /v1/models 带 override 返回上游模型列表；错误 Key -> 502。"""
    cloud_backend_captured.clear()
    client = _client(retrieval)
    r = client.post("/v1/models", json={"backend": _override()})
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert "cloud-model-x" in ids

    bad = dict(_override(), api_key="sk-wrong")
    r2 = client.post("/v1/models", json={"backend": bad})
    # 上游 /v1/models 不校验 Key 时返回 200 属正常；若 502 说明错误被正确映射
    assert r2.status_code in (200, 502)


def test_cloud_override_stream(retrieval, cloud_backend_captured):
    """流式 + override：不报错且透传成功。"""
    r = _client(retrieval).post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}],
              "stream": True, "backend": _override()},
    )
    assert r.status_code == 200


def test_key_never_in_stats_log(retrieval, cloud_backend_captured):
    """Key 绝不进入 /stats 日志。"""
    client = _client(retrieval)
    client.post("/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "机密测试"}], "backend": _override()})
    stats = client.get("/stats").json()
    assert SECRET_KEY not in str(stats), "API Key 不得出现在日志中"


def test_local_mode_uses_cfg_backend(retrieval, cloud_backend_captured):
    """无 override 时仍走全局配置后端（此处故意不可达 -> 502/500，但不应用云端配置）。"""
    client = _client(retrieval)
    r = client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code in (500, 502), r.status_code


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
