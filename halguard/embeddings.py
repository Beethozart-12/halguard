"""可插拔文本嵌入器。

- TFIDFEmbedder：纯 sklearn，零重依赖，适合本地轻量部署与测试。
- STEmbedder：sentence-transformers，语义质量更高（可选安装）。
get_embedder() 按配置返回合适的实现；auto 模式下若未安装 sentence-transformers 则回退 TF-IDF。
"""
from __future__ import annotations

import re
import pickle
from abc import ABC, abstractmethod

import numpy as np


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class Embedder(ABC):
    name = "base"
    _dim: int = 0

    @abstractmethod
    def embed(self, texts):
        ...

    def dim(self) -> int:
        return self._dim

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "Embedder":
        with open(path, "rb") as f:
            return pickle.load(f)


class TFIDFEmbedder(Embedder):
    name = "tfidf"

    def __init__(self):
        self._vec = None  # TfidfVectorizer
        self._dim = 0
        self._fitted = False

    def fit(self, docs):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._vec.fit(docs)
        self._dim = len(self._vec.vocabulary_)
        self._fitted = True
        return self

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if not self._fitted:
            # 未拟合时的兜底：哈希词袋
            return self._hash_embed(texts)
        X = self._vec.transform(texts)
        return np.asarray(X.todense(), dtype=np.float32)

    def _hash_embed(self, texts, dim: int = 1024):
        out = []
        for t in texts:
            v = np.zeros(dim, dtype=np.float32)
            for w in re.findall(r"\w+", t.lower()):
                v[hash(w) % dim] += 1.0
            out.append(v)
        return _l2_normalize(np.vstack(out)) if out else np.zeros((0, dim), dtype=np.float32)

    def __getstate__(self):
        return {"vec": self._vec, "dim": self._dim, "fitted": self._fitted}

    def __setstate__(self, state):
        self._vec = state["vec"]
        self._dim = state["dim"]
        self._fitted = state["fitted"]


class STEmbedder(Embedder):
    name = "st"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        # ST 3.x 用 get_sentence_embedding_dimension；ST 6.x 起改名为 get_embedding_dimension
        try:
            self._dim = self._model.get_embedding_dimension()
        except AttributeError:
            self._dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True), dtype=np.float32
        )

    def __getstate__(self):
        return {"model_name": self._model_name}

    def __setstate__(self, state):
        from sentence_transformers import SentenceTransformer
        self._model_name = state["model_name"]
        self._model = SentenceTransformer(self._model_name)
        # 与 __init__ 保持一致：ST 6.x 起改名 get_embedding_dimension
        try:
            self._dim = self._model.get_embedding_dimension()
        except AttributeError:
            self._dim = self._model.get_sentence_embedding_dimension()


def get_embedder(kind: str = "auto", st_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embedder:
    """按 kind 返回嵌入器。auto: 优先 ST，失败回退 TF-IDF。"""
    kind = (kind or "auto").lower()
    if kind in ("st", "sentence-transformers", "auto"):
        try:
            return STEmbedder(st_model)
        except Exception as e:  # 未安装或下载失败
            if kind == "st":
                raise
            # auto 回退
            print(f"[HalGuard] sentence-transformers 不可用，回退 TF-IDF 嵌入器: {e}")
    return TFIDFEmbedder()
