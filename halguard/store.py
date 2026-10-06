"""轻量向量库（纯 numpy 余弦相似度，无需 FAISS）。

生产环境可安装 faiss-cpu 替换（见 requirements-optional.txt），但默认实现零依赖即可运行。
"""
from __future__ import annotations

import pickle

import numpy as np


class VectorStore:
    def __init__(self, dim: int = 0):
        self.dim = dim
        self.texts: list[str] = []
        self.meta: list[dict] = []
        self.matrix: np.ndarray | None = None  # (n, dim) 已 L2 归一化

    def add(self, texts, embeddings, meta=None):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        # L2 归一化，使点积 == 余弦相似度
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        if self.matrix is None:
            self.matrix = embeddings
            self.dim = embeddings.shape[1]
        else:
            self.matrix = np.vstack([self.matrix, embeddings])
        self.texts.extend(texts)
        self.meta.extend(meta or [{} for _ in texts])

    def search(self, query_emb, top_k: int = 5):
        if self.matrix is None or len(self.texts) == 0:
            return [], [], []
        # query_emb 可能为 (1, D) 或 (D,)，规整为 1-D 以保证 matmul 维度正确
        q = np.asarray(query_emb, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(q)
        q = q / n if n > 0 else q
        M = self.matrix / np.linalg.norm(self.matrix, axis=1, keepdims=True)
        sims = M @ q  # (n,)
        k = min(top_k, len(sims))
        order = np.argsort(-sims)[:k]
        return order.tolist(), [float(sims[i]) for i in order], [self.texts[i] for i in order]

    def count(self) -> int:
        return len(self.texts)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {"dim": self.dim, "texts": self.texts, "meta": self.meta, "matrix": self.matrix},
                f,
            )

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        with open(path, "rb") as f:
            d = pickle.load(f)
        s = cls(d.get("dim", 0))
        s.texts = d.get("texts", [])
        s.meta = d.get("meta", [])
        s.matrix = d.get("matrix")
        return s
