"""离线比较上下文策略：证据有没有全部进入上下文（不调用大模型）。

    uv run python src/context_recall.py

证据完整的定义：
- 问题点名了产品：每条标注证据都要进上下文（对比题、跨条款题要全部拿到）；
- 没点名产品：标注的是行业规范和各产品的同名条款，任意一条进上下文就够。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from answer import build_context, select_seeds
from bm25 import BM25, RESOURCES, configure, tokenize
from evaluate import CHUNKS, EMB_DIR, GOLDEN, REPORTS, Retriever, detect_products, is_relevant, load_jsonl

SETS = [("golden_v2.jsonl", "bge-m3_index_text"), ("golden_blind_v1.jsonl", "bge-m3_index_text_blind")]
CONFIGS = {
    "基线（k5，扩展±1，8000 字）": dict(k=5, quota=0, neighbors=0, budget=8000),
    "k8": dict(k=8, quota=0, neighbors=0, budget=12000),
    "k10": dict(k=10, quota=0, neighbors=0, budget=16000),
    "对比题按产品分配 3 块": dict(k=5, quota=3, neighbors=0, budget=8000),
    "相邻条款（前 2 个种子）": dict(k=5, quota=0, neighbors=2, budget=12000),
    "分配 3 块 + 相邻条款": dict(k=5, quota=3, neighbors=2, budget=12000),
    "对比题按产品分配 5 块": dict(k=5, quota=5, neighbors=0, budget=12000),
    "相邻条款（全部 5 个种子）": dict(k=5, quota=0, neighbors=5, budget=16000),
    "分配 5 块 + 相邻条款（全部）": dict(k=5, quota=5, neighbors=5, budget=20000),
}


def evidence_complete(q: dict, ctx_chunks: list[dict]) -> tuple[bool, float]:
    srcs = q["expected_sources"]
    got = [any(is_relevant(c, s) for c in ctx_chunks) for s in srcs]
    need_all = bool(detect_products(q["question"]))
    return (all(got) if need_all else any(got)), sum(got) / len(srcs)


def main() -> None:
    configure(RESOURCES / "insurance_terms_v2.txt", mode="search")
    chunks = load_jsonl(CHUNKS)
    index = BM25([tokenize(c["index_text"]) for c in chunks])
    out = {}
    for golden_file, emb in SETS:
        golden = load_jsonl(GOLDEN.parent / golden_file)
        ret = Retriever("dense", index, np.load(EMB_DIR / emb / "chunks.npy"), np.load(EMB_DIR / emb / "questions.npy"))
        orders = []
        for qi, q in enumerate(golden):
            order, _ = ret.rank(qi, tokenize(q["question"]))
            products = detect_products(q["question"])
            if products:
                order = [i for i in order if chunks[i]["doc_id"] in products]
            orders.append((order, products))
        for name, cfg in CONFIGS.items():
            complete, recall, chars, missing = 0, 0.0, 0, []
            answerable = [(q, o) for q, o in zip(golden, orders) if q["type"] != "refuse"]
            for q, (order, products) in answerable:
                seeds = select_seeds(chunks, order, products, cfg["k"], cfg["quota"], cfg["neighbors"])
                ctx = build_context(chunks, seeds, len(seeds), cfg["budget"], expand=1)
                ctx_chunks = [chunks[j] for b in ctx for j in b["idxs"]]
                ok, rec = evidence_complete(q, ctx_chunks)
                complete += ok
                recall += rec
                chars += sum(c["n_chars"] for c in ctx_chunks)
                if not ok:
                    missing.append(q["id"])
            n = len(answerable)
            out.setdefault(name, {})[golden_file] = SimpleNamespace(
                complete=complete / n, recall=recall / n, chars=chars / n, missing=missing
            )
    lines = [
        "# 上下文证据完整率（离线，不调用模型）",
        "",
        "> 2026-09-26。检索 bge-m3 + 产品过滤；上下文扩展 ±1 块。只统计可回答题。",
        "",
        "| 策略 | golden_v2 完整率 | 证据召回 | 平均字数 | 盲测 完整率 | 证据召回 | 平均字数 | 盲测未完整的题 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in out.items():
        g, b = r["golden_v2.jsonl"], r["golden_blind_v1.jsonl"]
        lines.append(
            f"| {name} | {g.complete:.1%} | {g.recall:.1%} | {g.chars:.0f} | {b.complete:.1%} | {b.recall:.1%} | {b.chars:.0f} | {'、'.join(b.missing)} |"
        )
        print(f"{name:<20} golden {g.complete:.1%} ({g.chars:.0f}字) missing={g.missing} | blind {b.complete:.1%} ({b.chars:.0f}字) missing={b.missing}")
    (REPORTS / "context_recall_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
