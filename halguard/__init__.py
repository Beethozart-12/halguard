"""HalGuard — 本地 LLM/SLM 抗幻觉中间件。

让本地部署的 AI 在运行时通过检索接地 + 生成后验证来降低幻觉率。
"""
__version__ = "0.1.0"

from .config import HalGuardConfig
from .retrieval import RetrievalIndex, get_embedder  # noqa: F401
