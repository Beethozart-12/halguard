"""HalGuard 代理服务（FastAPI）。

对外暴露 OpenAI 兼容接口 /v1/chat/completions：
  应用/客户端把 base_url 指向本服务的地址即可，本地 LLM 在运行时本服务也随同工作。

流程：
  1. 检索：根据用户问题从本地知识库检索相关上下文（RAG 接地）
  2. 注入：把上下文与"仅依此作答"的指令注入 system 消息
  3. 生成：转发到后端本地 LLM
  4. 验证：抽取回答中的事实断言，对照上下文计算支持度，标记疑似幻觉
  5. 动作：按 action_on_risk 执行 flag（追加警告）/ redact（改写）/ reask（重问）/ none
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse

from .config import HalGuardConfig
from .retrieval import RetrievalIndex
from . import verify as V


def _last_user_message(messages: List[Dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):  # 多模态内容
                return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            return content or ""
    return ""


def _build_context_block(chunks: List[str]) -> Optional[str]:
    if not chunks:
        return None
    body = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
    return (
        "你是严谨的助手。请【仅依据】下面提供的[参考上下文]作答，"
        "不要引入上下文之外的任何事实；对关键事实用来源编号（如 [1]）标注。\n\n"
        "=== 参考上下文 ===\n" + body
    )


def _inject_context(messages: List[Dict], context_block: str) -> List[Dict]:
    out = [dict(m) for m in messages]
    if out and out[0].get("role") == "system":
        out[0] = dict(out[0])
        out[0]["content"] = (out[0].get("content", "") + "\n\n" + context_block).strip()
    else:
        out.insert(0, {"role": "system", "content": context_block})
    return out


async def _call_backend(client: httpx.AsyncClient, cfg: HalGuardConfig, payload: Dict) -> Dict:
    url = cfg.backend_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.backend_api_key}",
        "Content-Type": "application/json",
    }
    resp = await client.post(url, json=payload, headers=headers, timeout=180.0)
    resp.raise_for_status()
    return resp.json()


async def _extract_claims_llm(client, cfg, text, payload_template) -> List[str]:
    """可选：用后端 LLM 抽取断言（失败回退启发式）。"""
    sys_msg = {
        "role": "system",
        "content": "请从下面的回答中抽取所有事实性断言（fact claim），每条一行，不要解释，不要编号。若没有事实性断言，只回复空行。",
    }
    user_msg = {"role": "user", "content": text}
    p = dict(payload_template)
    p["messages"] = [sys_msg, user_msg]
    p["stream"] = False
    try:
        data = await _call_backend(client, cfg, p)
        content = data["choices"][0]["message"]["content"]
        return [s.strip("-•* ").strip() for s in content.splitlines() if s.strip()]
    except Exception:
        return V.extract_claims(text)


def build_app(cfg: HalGuardConfig, retrieval: RetrievalIndex) -> FastAPI:
    app = FastAPI(title="HalGuard - 本地 LLM 抗幻觉中间件", version="0.1.0")
    _log: List[Dict] = []
    _lock = threading.Lock()

    def _record(entry: Dict):
        with _lock:
            _log.append(entry)
            if len(_log) > 200:
                del _log[0]

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "backend": cfg.backend_base_url,
            "backend_model": cfg.backend_model,
            "kb_chunks": retrieval.count(),
            "embedder": retrieval.embedder.name,
        }

    @app.get("/v1/models")
    async def models():
        # 透传，便于客户端发现模型
        async with httpx.AsyncClient() as client:
            try:
                data = await _call_backend(client, cfg, {"model": cfg.backend_model, "messages": []})
            except Exception:
                data = {"object": "list", "data": [{"id": cfg.backend_model, "object": "model"}]}
        return JSONResponse(data)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload = await request.json()
        messages = payload.get("messages", [])
        query = _last_user_message(messages)

        # 检索上下文
        chunks, chunk_embs = ([], None)
        if retrieval.count() > 0 and query:
            chunks, chunk_embs = retrieval.query(query, top_k=cfg.top_k)

        # 流式：直接透传（不做验证，避免破坏流）
        if payload.get("stream"):
            url = cfg.backend_base_url.rstrip("/") + "/chat/completions"
            headers = {
                "Authorization": f"Bearer {cfg.backend_api_key}",
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(url, json=payload, headers=headers, timeout=180.0)
            return JSONResponse(resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text)

        # 构造转发 payload
        fwd = dict(payload)
        fwd["model"] = cfg.backend_model
        fwd["stream"] = False
        if chunks:
            fwd["messages"] = _inject_context(messages, _build_context_block(chunks))

        async with httpx.AsyncClient(timeout=180.0) as client:
            data = await _call_backend(client, cfg, fwd)
            content = data["choices"][0]["message"]["content"]

            # 仅在存在可对照上下文时才验证
            report: List[Dict] = []
            risk = 0.0
            action_taken = "none"
            if chunks:
                claims = (
                    await _extract_claims_llm(client, cfg, content, fwd)
                    if cfg.claim_extraction == "llm"
                    else V.extract_claims(content)
                )
                report = V.score_claims(claims, chunk_embs, retrieval.embedder, cfg.support_threshold, chunk_texts=chunks)
                risk = V.compute_risk(report)

                if risk > cfg.risk_threshold and cfg.action_on_risk == "reask":
                    # 重问：追加指令与不受支持的断言清单
                    reask_msgs = list(fwd["messages"])
                    reask_msgs.append(
                        {
                            "role": "system",
                            "content": cfg.reask_instruction
                            + "\n不受支持的断言：\n"
                            + "\n".join(f"- {r['claim']}" for r in report if r["unsupported"]),
                        }
                    )
                    reask_payload = dict(fwd)
                    reask_payload["messages"] = reask_msgs
                    data2 = await _call_backend(client, cfg, reask_payload)
                    content = data2["choices"][0]["message"]["content"]
                    data = data2
                    action_taken = "reask"

                elif risk > cfg.risk_threshold and cfg.action_on_risk == "redact":
                    content = V.redact(content, report)
                    data["choices"][0]["message"]["content"] = content
                    action_taken = "redact"

                elif risk > cfg.risk_threshold and cfg.action_on_risk == "flag":
                    action_taken = "flag"

        # 把警告块追加到回复（可选）
        warning = None
        if action_taken == "flag" and cfg.append_warning and report:
            warning = V.build_warning(report, risk)
            content = content + "\n\n" + warning
            data["choices"][0]["message"]["content"] = content

        # 附加 HalGuard 元数据（OpenAI 客户端通常忽略额外字段）
        data["halguard"] = {
            "risk": risk,
            "action": action_taken,
            "retrieved_chunks": len(chunks),
            "claims": report,
            "warning": warning,
        }

        _record({
            "ts": time.time(),
            "query": query[:200],
            "risk": risk,
            "action": action_taken,
            "unsupported": [r["claim"] for r in report if r["unsupported"]],
        })
        return JSONResponse(data)

    @app.get("/stats")
    def stats():
        with _lock:
            recent = list(_log)
        risks = [e["risk"] for e in recent]
        return {
            "total_checks": len(recent),
            "avg_risk": round(sum(risks) / len(risks), 4) if risks else 0.0,
            "recent": recent[-20:],
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard():
        with _lock:
            recent = list(_log)
        rows = "".join(
            f"<tr><td>{e['ts']:.0f}</td><td>{e['risk']:.2f}</td><td>{e['action']}</td>"
            f"<td>{e['query'].replace('<', '&lt;')}</td></tr>"
            for e in reversed(recent)
        )
        return HTMLResponse(
            "<h1>HalGuard 看板</h1>"
            "<p>每次本地 LLM 回答都会在此记录幻觉风险评分与触发的动作。</p>"
            "<table border='1' cellpadding='6'>"
            "<tr><th>时间</th><th>风险</th><th>动作</th><th>查询</th></tr>"
            f"{rows or '<tr><td colspan=4>暂无记录</td></tr>'}"
            "</table>"
        )

    return app
