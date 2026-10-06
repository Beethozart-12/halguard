import numpy as np
from halguard.store import VectorStore


def test_search_returns_topk_and_order():
    s = VectorStore(dim=3)
    s.add(["a", "b", "c"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    idx, scores, texts = s.search([1, 0, 0], top_k=2)
    assert texts[0] == "a"
    assert scores[0] >= scores[1]


def test_empty_store():
    s = VectorStore()
    idx, scores, texts = s.search([1, 0], top_k=3)
    assert texts == []


def test_save_load_roundtrip(tmp_path):
    s = VectorStore(dim=2)
    s.add(["x", "y"], [[1, 0], [0, 1]])
    p = str(tmp_path / "store.pkl")
    s.save(p)
    loaded = VectorStore.load(p)
    assert loaded.count() == 2
    assert loaded.search([1, 0], top_k=1)[2][0] == "x"
