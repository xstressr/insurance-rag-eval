"""拆子问题的检索诊断：不调用生成模型，只看证据有没有进上下文。

    uv run python src/split_recall.py --configs base,d1,d2 --sets blind3,blind5

只有拆分调用（不带思考，结果缓存）会花钱。指标：
- 任一证据：可回答题中，上下文里至少有一条标注证据；
- 全部证据：需要两条及以上证据的题中，每一条证据都在上下文里（单次 RAG 在多问题形态上输给 Agent 的主要原因）。
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import date

import numpy as np

from answer import LLM, NO_THINKING, retrieve
from bm25 import BM25, RESOURCES, configure, tokenize
from evaluate import CHUNKS, EMB_DIR, GOLDEN, REPORTS, Retriever, is_relevant, load_jsonl

SETS = {
    "golden_v2": ("golden_v2.jsonl", "bge-m3_index_text"),
    "blind1": ("golden_blind_v1.jsonl", "bge-m3_index_text_blind"),
    "blind2_1": ("golden_blind_v2.1.jsonl", "bge-m3_index_text_blind2"),
    "blind3": ("golden_blind_v3.jsonl", "bge-m3_index_text_blind3"),
    "blind5": ("golden_blind_v5.jsonl", "bge-m3_index_text_blind5"),
}
# 与 answer.py 的默认参数一致：k=5、expand=1、预算 8000；拆开时每个检索词取前 sub_k 块、预算 split_budget
BASE = dict(k=5, char_budget=8000, expand=1, quota=0, neighbors=0, whole_clause=0, sub_k=3, split_budget=12000, decompose=None)
CONFIGS = {
    "base": {},
    "d1": {"decompose": "d1"},
    "d2": {"decompose": "d2"},
    "d2_k5": {"decompose": "d2", "sub_k": 5},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="base,d1,d2")
    ap.add_argument("--sets", default="blind3,blind5")
    ap.add_argument("--name", default="split_recall")
    args = ap.parse_args()
    configure(RESOURCES / "insurance_terms_v2.txt", mode="search")
    chunks = load_jsonl(CHUNKS)
    index = BM25([tokenize(c["index_text"]) for c in chunks])
    splitter = LLM(None, extra_body=NO_THINKING)

    lines = [
        f"# 拆子问题的检索诊断 · {args.name}",
        "",
        f"> 自动生成（{date.today()}），`src/split_recall.py`。不调用生成模型，只看标注证据有没有进上下文。",
        "",
        "| 集合 | 配置 | 任一证据 | 全部证据（多证据题） | 平均上下文字数 | 走多查询的题 |",
        "|---|---|---|---|---|---|",
    ]
    misses = []
    for set_name in args.sets.split(","):
        golden_file, emb_name = SETS[set_name]
        golden = load_jsonl(GOLDEN.parent / golden_file)
        emb = EMB_DIR / emb_name
        retriever = Retriever("dense", index, np.load(emb / "chunks.npy"), np.load(emb / "questions.npy"))
        for cfg_name in args.configs.split(","):
            cfg = argparse.Namespace(**{**BASE, **CONFIGS[cfg_name]}, run_name=f"gen_v4_{cfg_name}_{set_name}")
            any_hit = full = n_ans = n_multi = n_split = chars = 0
            for qi, q in enumerate(golden):
                session = str(uuid.uuid5(uuid.NAMESPACE_URL, f"insurance-rag-eval/{cfg.run_name}/{q['id']}"))
                ctx, _, subs, _ = retrieve(q, qi, chunks, retriever, cfg, session, splitter if cfg.decompose else None)
                in_ctx = [chunks[j] for b in ctx for j in b["idxs"]]
                chars += sum(c["n_chars"] for c in in_ctx)
                n_split += len(subs) >= (2 if cfg.decompose == "d1" else 1) and bool(cfg.decompose)
                srcs = q["expected_sources"]
                if q["type"] == "refuse" or not srcs:
                    continue
                hit = [any(is_relevant(c, s) for c in in_ctx) for s in srcs]
                n_ans += 1
                any_hit += any(hit)
                if len(srcs) >= 2:
                    n_multi += 1
                    full += all(hit)
                    if not all(hit):
                        misses.append(f"| {set_name} | {cfg_name} | {q['id']} | {'；'.join(subs) or '—'} |")
            lines.append(
                f"| {set_name} | {cfg_name} | {any_hit}/{n_ans} | {full}/{n_multi} | {chars / len(golden):.0f} | "
                f"{n_split}/{len(golden) if cfg.decompose else 0} |"
            )
    lines += ["", "## 多证据题没找全的", "", "| 集合 | 配置 | 题 | 子问题 / 检索词 |", "|---|---|---|---|", *misses, ""]
    out = REPORTS / f"{args.name}.md"
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("\n".join(l for l in lines if l.startswith("| blind") or l.startswith("| golden") or l.startswith("| 集合 | 配置 | 任")))


if __name__ == "__main__":
    main()
