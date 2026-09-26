"""加权 RRF 网格搜索，只在调参集（默认 golden_v1）上运行，不写报告。

    python src/sweep_fusion.py

选参规则（运行前定死）：先比 Hit@5，再比 MRR@10；打平时优先 rrf_k=60、BM25 权重较小者。
"""

from __future__ import annotations

import json

import numpy as np

from bm25 import BM25, configure, tokenize
from evaluate import CHUNKS, EMB_DIR, GOLDEN, RESOURCES, Retriever, evaluate, load_jsonl, summarize

WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
RRF_KS = [20, 60]


def main() -> None:
    configure(RESOURCES / "insurance_terms_v2.txt", mode="search")
    chunks, golden = load_jsonl(CHUNKS), load_jsonl(GOLDEN)
    emb = EMB_DIR / "bge-m3_index_text"
    meta = json.loads((emb / "meta.json").read_text(encoding="utf-8"))
    assert meta["question_ids"] == [q["id"] for q in golden]
    cv, qv = np.load(emb / "chunks.npy"), np.load(emb / "questions.npy")
    index = BM25([tokenize(c["index_text"]) for c in chunks])

    rows = []
    for k in RRF_KS:
        for w in WEIGHTS:
            res = evaluate(chunks, golden, Retriever("hybrid", index, cv, qv, rrf_k=k, w_bm25=w), 5, product_filter=True)
            s = summarize([r for r in res if r["type"] != "refuse"])
            miss = [r["id"] for r in res if r["type"] != "refuse" and not r["hit"]]
            rows.append((k, w, s["hit"], s["mrr"], miss))
            print(f"rrf_k={k:>2} w_bm25={w:.1f}  Hit@5={s['hit']:.1%}  MRR@10={s['mrr']:.3f}  miss={miss}")

    best = sorted(rows, key=lambda r: (-r[2], -round(r[3], 3), r[0] != 60, r[1]))[0]
    print(f"\n选中：rrf_k={best[0]} w_bm25={best[1]}")


if __name__ == "__main__":
    main()
