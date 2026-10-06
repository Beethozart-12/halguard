"""检索索引：知识库摄入（chunk -> 嵌入 -> 向量库）与查询。

持久化到 index_path 目录：embedder.pkl（含 TF-IDF 词表）+ store.pkl（向量库）。
"""
from __future__ import annotations

import os
import glob
from typing import List, Tuple

import numpy as np

from .embeddings import Embedder, get_embedder
from .store import VectorStore

CHUNK_SIZE = 480
CHUNK_OVERLAP = 80
SUPPORTED_EXT = (".txt", ".md", ".json")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # 优先按段落，再按字符窗口
    paras = [p for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    for p in paras:
        if len(p) <= size:
            chunks.append(p.strip())
            continue
        start = 0
        while start < len(p):
            end = min(start + size, len(p))
            chunks.append(p[start:end].strip())
            if end == len(p):
                break
            start = max(end - overlap, start + 1)
    return [c for c in chunks if len(c) >= 12]


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


class RetrievalIndex:
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    # ---------- 摄入 ----------
    def ingest(self, kb_path: str, force_refit: bool = False) -> int:
        files: List[str] = []
        if os.path.isfile(kb_path):
            files = [kb_path]
        else:
            for ext in SUPPORTED_EXT:
                files.extend(glob.glob(os.path.join(kb_path, "**", f"*{ext}"), recursive=True))
        if not files:
            raise RuntimeError(f"在 {kb_path} 未找到可摄入的文本（{SUPPORTED_EXT}）。")

        chunks: List[str] = []
        for fp in files:
            try:
                chunks.extend(chunk_text(_read_file(fp)))
            except Exception as e:
                print(f"[HalGuard] 跳过文件 {fp}: {e}")
        if not chunks:
            raise RuntimeError("知识库为空，未生成任何文本块。")

        # TF-IDF 需要先在整个语料上拟合
        if self.embedder.name == "tfidf" and (not getattr(self.embedder, "_fitted", False) or force_refit):
            self.embedder.fit(chunks)

        embs = self.embedder.embed(chunks)
        self.store.add(chunks, embs, meta=[{"src": "kb"} for _ in chunks])
        return len(chunks)

    # ---------- 查询 ----------
    def query(self, q: str, top_k: int = 5) -> Tuple[List[str], np.ndarray]:
        if self.store.count() == 0:
            return [], np.zeros((0, self.embedder.dim()), dtype=np.float32)
        q_emb = self.embedder.embed(q)
        idx, scores, texts = self.store.search(q_emb, top_k=top_k)
        if len(idx) == 0:
            return [], np.zeros((0, self.embedder.dim()), dtype=np.float32)
        chunk_embs = self.store.matrix[idx]
        return texts, chunk_embs

    def count(self) -> int:
        return self.store.count()

    # ---------- 持久化 ----------
    def save(self, index_path: str):
        os.makedirs(index_path, exist_ok=True)
        self.embedder.save(os.path.join(index_path, "embedder.pkl"))
        self.store.save(os.path.join(index_path, "store.pkl"))

    @classmethod
    def load(cls, index_path: str, embedder_kind: str = "auto", st_model: str = "") -> "RetrievalIndex":
        emb_path = os.path.join(index_path, "embedder.pkl")
        store_path = os.path.join(index_path, "store.pkl")
        if not (os.path.exists(emb_path) and os.path.exists(store_path)):
            raise FileNotFoundError(
                f"未找到索引文件（{emb_path} / {store_path}）。请先运行 `halguard ingest`。"
            )
        embedder = Embedder.load(emb_path)
        store = VectorStore.load(store_path)
        return cls(embedder, store)

    @classmethod
    def new(cls, embedder_kind: str = "auto", st_model: str = "") -> "RetrievalIndex":
        embedder = get_embedder(embedder_kind, st_model) if st_model else get_embedder(embedder_kind)
        store = VectorStore(embedder.dim())
        return cls(embedder, store)
