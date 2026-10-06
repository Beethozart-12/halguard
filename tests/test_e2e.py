"""HalGuard 端到端集成测试。

真实链路：mock 后端（模拟会幻觉的本地 LLM）-> HalGuard 代理 -> 客户端请求。
验证：检索接地注入、生成后验证、幻觉标记、flag 动作。

该测试依赖 sentence-transformers 与本地多语种模型；若模型缺失则跳过（不视为失败）。
"""
import os
import sys
import time
import threading
import tempfile
import textwrap

import pytest

# 本地多语种模型路径（位于工作区根目录的 halguard_models/，即仓库的上两级）
_HERE = os.path.dirname(__file__)
MODEL = os.path.abspath(os.path.join(_HERE, "..", "..", "halguard_models", "paraphrase-multilingual-MiniLM-L12-v2"))
HAS_MODEL = os.path.isdir(MODEL)

pytestmark = pytest.mark.skipif(
    not HAS_MODEL,
    reason="本地多语种模型未找到（halguard_models/paraphrase-multilingual-MiniLM-L12-v2），跳过端到端测试",
)

import httpx
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import uvicorn

from halguard.config import HalGuardConfig
from halguard.retrieval import RetrievalIndex
from halguard.server import build_app


def _make_backend():
    """构造一个会幻觉的 mock 后端，并返回 (app, 捕获请求列表)。"""
    captured = []

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request):
        body = await req.json()
        captured.append(body)
        # 故意返回含幻觉的回答：前半句"巴黎是德国的首都"与知识库矛盾
        return {
            "id": "mock-1",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "巴黎是德国的首都。巴黎是法国的首都，位于塞纳河畔。",
                    },
                    "finish_reason": "stop",
                }
            ],
            "model": "mock",
        }

    return app, captured


def _wait_until_ready(url: str, timeout: float = 20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            httpx.get(url, timeout=1.0)
            return True
        except Exception:
            time.sleep(0.3)
    return False


@pytest.fixture(scope="module")
def backend_url():
    app, _ = _make_backend()
    cfg = uvicorn.Config(app, host="127.0.0.1", port=18923, log_level="warning")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    assert _wait_until_ready("http://127.0.0.1:18923/"), "mock 后端未能启动"
    yield "http://127.0.0.1:18923/v1"
    server.should_exit = True


@pytest.fixture(scope="module")
def retrieval():
    # 受控知识库：巴黎是法国首都 / 柏林是德国首都
    kb = tempfile.mkdtemp(prefix="hg_e2e_kb_")
    with open(os.path.join(kb, "paris.txt"), "w", encoding="utf-8") as f:
        f.write(
            textwrap.dedent(
                """\
                巴黎是法国的首都，位于塞纳河畔。埃菲尔铁塔在巴黎。
                巴黎是法国人口最多的城市，也是法兰西岛大区的首府。
                """
            )
        )
    idx_dir = tempfile.mkdtemp(prefix="hg_e2e_idx_")
    ri = RetrievalIndex.new(embedder_kind="st", st_model=MODEL)
    ri.ingest(kb)
    # 也持久化，验证 save/load 链路
    ri.save(idx_dir)
    ri2 = RetrievalIndex.load(idx_dir, embedder_kind="st", st_model=MODEL)
    return ri2


def test_e2e_grounding_and_hallucination_flag(backend_url, retrieval):
    cfg = HalGuardConfig(
        backend_base_url=backend_url,
        backend_api_key="mock",
        backend_model="mock",
        embedder="st",
        st_model=MODEL,
        action_on_risk="flag",
        claim_extraction="heuristic",
        support_threshold=0.35,
        risk_threshold=0.30,
        append_warning=True,
    )
    app = build_app(cfg, retrieval)
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock",
            "messages": [{"role": "user", "content": "巴黎的首都是哪里？"}],
            "stream": False,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    # 1) 后端收到了注入的接地上下文（RAG 接地生效）
    #    HalGuard 内部会调用 mock，这里通过再发一次请求无法拿到；改为校验 halguard 元数据
    #    与 grounding 注入：用独立探针发一个请求，检查 mock 收到的 system 消息。
    assert "halguard" in data, "响应缺少 halguard 元数据"
    hg = data["halguard"]

    # 2) 风险 > 0（存在不受支持断言）
    assert hg["risk"] > 0.0, f"期望风险>0，实际 {hg['risk']}"

    # 3) 幻觉被标记：至少一条不受支持断言，且包含"德国"
    unsupported = [r["claim"] for r in hg["claims"] if r["unsupported"]]
    assert unsupported, f"未标记任何疑似幻觉断言；claims={hg['claims']}"
    assert any("德国" in c for c in unsupported), f"未把'巴黎是德国的首都'标记为幻觉：{unsupported}"

    # 4) flag 动作触发
    assert hg["action"] == "flag", f"期望 flag 动作，实际 {hg['action']}"

    # 5) flag 时警告块被追加到回复末尾
    assert hg["warning"], "flag 动作应附带警告块"


def test_e2e_reask_on_high_risk(backend_url, retrieval):
    """reask 模式：高风险时向后端追加重问指令。"""
    cfg = HalGuardConfig(
        backend_base_url=backend_url,
        backend_api_key="mock",
        backend_model="mock",
        embedder="st",
        st_model=MODEL,
        action_on_risk="reask",
        claim_extraction="heuristic",
        support_threshold=0.35,
        risk_threshold=0.10,  # 低阈值，确保触发
    )
    app = build_app(cfg, retrieval)
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock",
            "messages": [{"role": "user", "content": "巴黎的首都是哪里？"}],
            "stream": False,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    hg = data["halguard"]
    assert hg["action"] == "reask", f"期望 reask 动作，实际 {hg['action']}"


def test_e2e_grounding_injected_into_backend(backend_url, retrieval):
    """探针：确认 mock 后端确实收到了含[参考上下文]的 system 消息。"""
    cfg = HalGuardConfig(
        backend_base_url=backend_url,
        backend_api_key="mock",
        backend_model="mock",
        embedder="st",
        st_model=MODEL,
        action_on_risk="none",
        claim_extraction="heuristic",
    )
    app = build_app(cfg, retrieval)
    client = TestClient(app)
    # 用一个捕获 backend 调用的方式：直接 patch _call_backend 不可行（已联网），
    # 改为启动第二套带捕获的 backend。
    captured = []

    bapp = FastAPI()

    @bapp.post("/v1/chat/completions")
    async def cc(req: Request):
        b = await req.json()
        captured.append(b)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "巴黎是德国的首都。巴黎是法国的首都，位于塞纳河畔。",
                    }
                }
            ],
            "model": "mock",
        }

    bcfg = uvicorn.Config(bapp, host="127.0.0.1", port=18924, log_level="warning")
    bserver = uvicorn.Server(bcfg)
    bt = threading.Thread(target=bserver.run, daemon=True)
    bt.start()
    assert _wait_until_ready("http://127.0.0.1:18924/")
    cfg2 = HalGuardConfig(
        backend_base_url="http://127.0.0.1:18924/v1",
        backend_api_key="mock",
        backend_model="mock",
        embedder="st",
        st_model=MODEL,
        action_on_risk="none",
        claim_extraction="heuristic",
    )
    app2 = build_app(cfg2, retrieval)
    c2 = TestClient(app2)
    r = c2.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "巴黎的首都是哪里？"}]},
    )
    assert r.status_code == 200
    assert captured, "mock 后端未收到请求"
    sys_msgs = [
        m for m in captured[0]["messages"] if m.get("role") == "system"
    ]
    assert sys_msgs, "后端未收到 system 消息"
    assert "参考上下文" in sys_msgs[0]["content"], "接地上下文未注入 system 消息"
    bserver.should_exit = True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
