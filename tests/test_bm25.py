"""用 rank_bm25 作为参考实现，验证手写 BM25 的打分完全一致。"""

import json
import sys
from pathlib import Path

import pytest
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bm25 import BM25, tokenize  # noqa: E402

CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
GOLDEN = ROOT / "dataset" / "golden_v1.jsonl"


def test_tokenize_keeps_insurance_terms():
    toks = tokenize("太保阿基米德的犹豫期是多少天？")
    assert "犹豫期" in toks
    assert "的" not in toks and "？" not in toks


def test_matches_rank_bm25_on_toy_corpus():
    docs = [["犹豫期", "15", "日"], ["等待期", "90", "日"], ["犹豫期", "解除", "合同", "合同"], ["现金价值"]]
    ours = BM25(docs, idf_variant="okapi")
    ref = BM25Okapi(docs)
    for q in [["犹豫期"], ["合同", "日"], ["等待期", "等待期"]]:
        assert ours.scores(q) == pytest.approx(list(ref.get_scores(q)))


@pytest.mark.skipif(not CHUNKS.exists(), reason="先运行 src/ingest.py 生成 chunks")
def test_matches_rank_bm25_on_real_corpus():
    chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
    docs = [tokenize(c["index_text"]) for c in chunks]
    ours = BM25(docs, idf_variant="okapi")
    ref = BM25Okapi(docs)
    for line in GOLDEN.open(encoding="utf-8"):
        q = tokenize(json.loads(line)["question"])
        assert ours.scores(q) == pytest.approx(list(ref.get_scores(q)), rel=1e-9, abs=1e-9)
