import os
from halguard.retrieval import RetrievalIndex


def test_ingest_query_roundtrip(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "a.txt").write_text("Paris is the capital of France. The Eiffel Tower is in Paris.")
    retr = RetrievalIndex.new(embedder_kind="tfidf")
    n = retr.ingest(str(kb))
    assert n > 0
    chunks, _ = retr.query("capital of France", top_k=2)
    assert any("Paris" in c for c in chunks)


def test_save_load_index(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "a.txt").write_text("Paris is the capital of France.")
    retr = RetrievalIndex.new(embedder_kind="tfidf")
    retr.ingest(str(kb))
    idx = str(tmp_path / ".idx")
    retr.save(idx)
    loaded = RetrievalIndex.load(idx, embedder_kind="tfidf")
    chunks, _ = loaded.query("capital of France", top_k=2)
    assert any("Paris" in c for c in chunks)
