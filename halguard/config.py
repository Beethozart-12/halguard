"""HalGuard 配置（零依赖，使用 dataclasses）。

所有字段都有合理默认值，可通过环境变量 HALGUARD_* 覆盖。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class HalGuardConfig:
    # ---- 后端本地 LLM（OpenAI 兼容接口，如 Ollama / LM Studio / vLLM）----
    backend_base_url: str = "http://localhost:11434/v1"   # Ollama 默认
    backend_api_key: str = "ollama"
    backend_model: str = "llama3.1:8b"

    # ---- 代理监听 ----
    listen_host: str = "0.0.0.0"
    listen_port: int = 8849

    # ---- 嵌入与检索 ----
    # embedder: auto(优先 sentence-transformers，否则 TF-IDF) | st | tfidf
    embedder: str = "auto"
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5

    # ---- 验证与策略 ----
    support_threshold: float = 0.35   # 单条断言语义支持度低于此值 -> 视为不受支持（疑似幻觉）
    risk_threshold: float = 0.30     # 整体风险（不受支持断言占比）高于此值 -> 触发动作
    action_on_risk: str = "flag"      # flag | redact | reask | none
    claim_extraction: str = "heuristic"  # heuristic | llm
    append_warning: bool = True       # 当 flag 时，把警告块追加到回复末尾

    # ---- 路径 ----
    kb_path: str = "./knowledge"
    index_path: str = "./.halguard_index"

    # ---- 重问（reask）时的附加指令 ----
    reask_instruction: str = (
        "你上一版的回答包含缺乏依据的断言。请只使用下面提供的[参考上下文]作答，"
        "不要引入上下文之外的任何事实，并对关键事实标注来源编号（如 [1]）。"
    )

    @staticmethod
    def load(prefix: str = "HALGUARD_") -> "HalGuardConfig":
        """从环境变量 HALGUARD_* 读取覆盖值。"""
        cfg = HalGuardConfig()
        mapping = {
            "BACKEND_BASE_URL": "backend_base_url",
            "BACKEND_API_KEY": "backend_api_key",
            "BACKEND_MODEL": "backend_model",
            "LISTEN_HOST": "listen_host",
            "LISTEN_PORT": "listen_port",
            "EMBEDDER": "embedder",
            "ST_MODEL": "st_model",
            "TOP_K": "top_k",
            "SUPPORT_THRESHOLD": "support_threshold",
            "RISK_THRESHOLD": "risk_threshold",
            "ACTION_ON_RISK": "action_on_risk",
            "CLAIM_EXTRACTION": "claim_extraction",
            "APPEND_WARNING": "append_warning",
            "KB_PATH": "kb_path",
            "INDEX_PATH": "index_path",
        }
        for env_key, attr in mapping.items():
            val = os.environ.get(prefix + env_key)
            if val is None:
                continue
            cur = getattr(cfg, attr)
            if isinstance(cur, bool):
                setattr(cfg, attr, val.lower() in ("1", "true", "yes", "on"))
            elif isinstance(cur, int):
                setattr(cfg, attr, int(val))
            elif isinstance(cur, float):
                setattr(cfg, attr, float(val))
            else:
                setattr(cfg, attr, val)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)
