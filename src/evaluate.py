"""在 golden 评测集上评估 BM25 检索，输出逐题结果和 Markdown 报告。

相关性判定：文本块与标注证据的文档、部分、条款号一致，且标注页码落在块的页码范围内。

指标（只统计有证据的题，拒答题单独列出）：
- Hit@k：前 k 名里至少有一个相关块。
- Recall@k：标注的证据有多少进了前 k（对比题有多条证据时有意义）。
- MRR@10：第一个相关块排名的倒数；前 10 名都没有记 0。

用法：uv run python src/evaluate.py --run-name bm25_v1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bm25 import BM25, RESOURCES, configure, tokenize

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
GOLDEN = ROOT / "dataset" / "golden_v1.jsonl"
REPORTS = ROOT / "reports"
EMB_DIR = ROOT / "data" / "processed" / "emb"

SHORT = {
    "cpic_archimedes_2025": "太保",
    "taikang_huijiabao_2026": "泰康",
    "chinalife_kangning_zunxiang_2024": "国寿",
    "iac_ci_definitions_2020": "行业规范",
}
RETRIEVER_NAME = {
    "bm25": "手写 BM25（已用 rank_bm25 对照验证）",
    "dense": "向量检索（余弦相似度）",
    "hybrid": "混合检索：BM25 + 向量，RRF 融合",
    "rerank": "向量检索粗选 + bge-reranker-v2-m3 精排",
}
TYPE_NAME = {"exact": "精确查找", "number": "数字", "condition": "条件判断", "compare": "对比", "refuse": "拒答"}

# 问题里出现这些别名，就认为在问对应文档。
PRODUCT_ALIASES = {
    "cpic_archimedes_2025": ["太保", "阿基米德", "太平洋"],
    "taikang_huijiabao_2026": ["泰康", "惠嘉保"],
    "chinalife_kangning_zunxiang_2024": ["国寿", "康宁尊享", "中国人寿"],
    "iac_ci_definitions_2020": ["行业规范", "疾病定义使用规范", "行业协会"],
}


def detect_products(question: str) -> set[str]:
    """识别问题提到的文档；一个都没提到时返回空集，表示不过滤。"""
    return {doc for doc, aliases in PRODUCT_ALIASES.items() if any(a in question for a in aliases)}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def is_relevant(chunk: dict, src: dict) -> bool:
    return (
        chunk["doc_id"] == src["doc_id"]
        and chunk["part"] == src["part"]
        and chunk["clause_id"] == src["clause_id"]
        and chunk["page_start"] <= src["page"] <= chunk["page_end"]
    )


def label(chunk: dict) -> str:
    where = "阅读指引" if chunk["section"] == "guide" else f"{chunk['part']} {chunk['clause_id']} {chunk['title']}".strip()
    return f"{SHORT[chunk['doc_id']]} p{chunk['page_start']} {where}"


class Retriever:
    """三种检索方式：bm25、dense（向量余弦）、hybrid（两路名次用 RRF 融合）。

    RRF：score = Σ 1 / (rrf_k + 名次)。只看名次、不看原始分数，
    所以 BM25 分数（几到几十）和余弦（0 到 1）量纲不同也能直接合并。
    加权 RRF：BM25 那一项乘以 w_bm25（Dense 固定为 1）。w_bm25=1 就是等权，=0 退化成纯 Dense。
    """

    def __init__(
        self,
        kind: str,
        index: BM25,
        chunk_vecs=None,
        question_vecs=None,
        rrf_k: int = 60,
        w_bm25: float = 1.0,
        rerank_scores=None,
        rerank_top: int = 0,
    ):
        self.kind, self.index, self.rrf_k, self.w_bm25 = kind, index, rrf_k, w_bm25
        # rerank：向量检索前 rerank_top 名按重排分数重新排序，其余保持向量顺序接在后面；0 = 全部重排
        self.rerank_scores, self.rerank_top = rerank_scores, rerank_top
        self.chunk_vecs, self.question_vecs = chunk_vecs, question_vecs

    def rank(self, qi: int, qtok: list[str]) -> tuple[list[int], dict[int, dict]]:
        """返回全部文本块的排序，以及每块的得分信息（用于报告里的解释）。"""
        n = self.index.n_docs
        bm = self.index.scores(qtok)
        bm_order = sorted(range(n), key=lambda i: (-bm[i], i))
        info = {i: {"bm25": round(bm[i], 2)} for i in range(n)}
        if self.kind == "bm25":
            for i in range(n):
                info[i]["score"] = bm[i]
            return bm_order, info

        cos = self.chunk_vecs @ self.question_vecs[qi]
        dense_order = sorted(range(n), key=lambda i: (-cos[i], i))
        bm_rank = {d: r for r, d in enumerate(bm_order, start=1)}
        dense_rank = {d: r for r, d in enumerate(dense_order, start=1)}
        for i in range(n):
            info[i].update(cos=round(float(cos[i]), 3), bm25_rank=bm_rank[i], dense_rank=dense_rank[i])
        if self.kind == "dense":
            for i in range(n):
                info[i]["score"] = float(cos[i])
            return dense_order, info
        if self.kind == "rerank":
            rr = self.rerank_scores[qi]
            top = dense_order[: self.rerank_top] if self.rerank_top else dense_order
            head = sorted(top, key=lambda i: (-rr[i], i))
            order = head + dense_order[len(top):]
            for r, i in enumerate(order, start=1):
                info[i].update(score=float(rr[i]) if i in set(top) else float("-inf"), rerank=round(float(rr[i]), 2))
            return order, info

        fused = [self.w_bm25 / (self.rrf_k + bm_rank[i]) + 1 / (self.rrf_k + dense_rank[i]) for i in range(n)]
        for i in range(n):
            info[i]["score"] = fused[i]
        return sorted(range(n), key=lambda i: (-fused[i], i)), info


def evaluate(chunks: list[dict], golden: list[dict], retriever: Retriever, k: int, product_filter: bool = False) -> list[dict]:
    index = retriever.index
    results = []
    for qi, q in enumerate(golden):
        qtok = tokenize(q["question"])
        order, info = retriever.rank(qi, qtok)
        products = detect_products(q["question"]) if product_filter else set()
        if products:  # 先排序再过滤：各块得分互相独立，保持相对顺序，等价于只在这些产品里检索
            order = [i for i in order if chunks[i]["doc_id"] in products]
        pos = {doc: r for r, doc in enumerate(order, start=1)}

        # 每条证据的最佳排名（一条证据可能对应同一条款的多个块）
        src_ranks = []
        for src in q["expected_sources"]:
            ranks = [pos[i] for i, c in enumerate(chunks) if is_relevant(c, src)]
            src_ranks.append(min(ranks) if ranks else None)

        found = [r for r in src_ranks if r is not None]
        first = min(found) if found else None
        top = order[:k]
        results.append(
            {
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "products": sorted(products),
                "query_tokens": qtok,
                "source_ranks": src_ranks,
                "first_rank": first,
                "hit": bool(first and first <= k),
                "recall": (sum(1 for r in found if r <= k) / len(src_ranks)) if src_ranks else None,
                "rr": (1 / first) if first and first <= 10 else 0.0,
                "top": [
                    {
                        "chunk_id": chunks[i]["chunk_id"],
                        "label": label(chunks[i]),
                        "section": chunks[i]["section"],
                        "relevant": any(is_relevant(chunks[i], s) for s in q["expected_sources"]),
                        "explain": (
                            {t: round(v, 2) for t, v in index.explain(qtok, i).items()}
                            if retriever.kind == "bm25"
                            else {x: v for x, v in info[i].items() if x != "score"}
                        ),
                    }
                    for i in top
                ],
                "top1_score": round(info[top[0]]["score"], 4),
            }
        )
    return results


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        "hit": sum(r["hit"] for r in rows) / n,
        "recall": sum(r["recall"] for r in rows) / n,
        "mrr": sum(r["rr"] for r in rows) / n,
    }


def write_report(results: list[dict], args: argparse.Namespace, n_chunks: int) -> Path:
    k = args.k
    answerable = [r for r in results if r["type"] != "refuse"]
    refuse = [r for r in results if r["type"] == "refuse"]
    overall = summarize(answerable)
    guide_slots = sum(t["section"] == "guide" for r in answerable for t in r["top"])

    lines = [
        f"# 检索评测报告 · {args.run_name}",
        "",
        f"> 自动生成。评测集 `dataset/{args.golden}`（{len(results)} 题，{args.golden_status}）。",
        "",
        "## 配置",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 检索器 | {RETRIEVER_NAME[args.retriever]}{'（向量：' + args.emb + '）' if args.retriever != 'bm25' else ''} |",
        *([f"| 重排 | 分数 `{args.rerank}`，向量检索前 {args.rerank_top or '全部'} 名进入重排 |"] if args.retriever == "rerank" else []),
        *([f"| RRF | rrf_k={args.rrf_k}，BM25 权重 {args.w_bm25}，Dense 权重 1 |"] if args.retriever == "hybrid" else []),
        f"| k1 / b / IDF | {args.k1} / {args.b} / {args.idf} |",
        f"| 分词 | jieba {'搜索引擎' if args.tokenizer == 'search' else '精确'}模式 + 词典 `{args.terms}` + 停用词 |",
        f"| 索引字段 | `{args.field}` |",
        f"| 产品过滤 | {'开（问题提到的产品才参与排序）' if args.product_filter else '关'} |",
        f"| 文本块数 | {n_chunks} |",
        f"| k | {k} |",
        "",
        f"## 总体指标（有证据的 {len(answerable)} 题）",
        "",
        f"| Hit@{k} | Recall@{k} | MRR@10 | 前 {k} 名中阅读指引占比 |",
        "|---|---|---|---|",
        f"| {overall['hit']:.1%} | {overall['recall']:.1%} | {overall['mrr']:.3f} | "
        f"{guide_slots / (len(answerable) * k):.1%} |",
        "",
        "## 分题型",
        "",
        f"| 题型 | 题数 | Hit@{k} | Recall@{k} | MRR@10 |",
        "|---|---|---|---|---|",
    ]
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in answerable:
        by_type[r["type"]].append(r)
    for t in ["exact", "number", "condition", "compare"]:
        if not by_type[t]:  # 有的评测集没有某类题（blind_v3 没有数字题）
            continue
        s = summarize(by_type[t])
        lines.append(f"| {TYPE_NAME[t]} | {s['n']} | {s['hit']:.1%} | {s['recall']:.1%} | {s['mrr']:.3f} |")

    lines += [
        "",
        "## 逐题结果",
        "",
        "| ID | 题型 | 问题 | 证据排名 | 命中 |",
        "|---|---|---|---|---|",
    ]
    for r in answerable:
        ranks = " / ".join(str(x) if x else "无" for x in r["source_ranks"])
        lines.append(f"| {r['id']} | {TYPE_NAME[r['type']]} | {r['question']} | {ranks} | {'✅' if r['hit'] else '❌'} |")

    lines += ["", f"## 未命中的题（前 {k} 名没有相关块）", ""]
    misses = [r for r in answerable if not r["hit"]]
    if not misses:
        lines.append("无。")
    for r in misses:
        lines += [
            f"### {r['id']} · {r['question']}",
            "",
            f"- 分词：`{' / '.join(r['query_tokens'])}`",
            f"- 证据实际排名：{' / '.join(str(x) if x else '无' for x in r['source_ranks'])}",
            "- 前 3 名及得分来源：",
        ]
        for t in r["top"][:3]:
            lines.append(f"  - {t['label']}：{t['explain']}")
        lines.append("")

    partial = [r for r in answerable if r["hit"] and r["recall"] < 1]
    if partial:
        lines += ["## 部分命中的对比题", ""]
        for r in partial:
            lines.append(f"- {r['id']}：证据排名 {r['source_ranks']}，Recall@{k} = {r['recall']:.0%}")
        lines.append("")

    lines += [
        "## 拒答题（检索阶段不计分，记录第 1 名得分供后续设阈值）",
        "",
        "| ID | 问题 | 第 1 名 | 得分 |",
        "|---|---|---|---|",
    ]
    for r in refuse:
        lines.append(f"| {r['id']} | {r['question']} | {r['top'][0]['label']} | {r['top1_score']} |")
    answer_top1 = sorted(r["top1_score"] for r in answerable)
    lines += [
        "",
        f"对照：有证据题的第 1 名得分中位数 {answer_top1[len(answer_top1) // 2]}，"
        f"最低 {answer_top1[0]}。",
        "",
        "## 结论",
        "",
    ]
    out = REPORTS / f"{args.run_name}.md"
    # 重跑时保留学习者已写的结论，只刷新自动生成的部分
    old = out.read_text(encoding="utf-8").split("## 结论\n", 1) if out.exists() else []
    conclusion = old[1].strip() if len(old) == 2 else ""
    lines.append(conclusion or "（由学习者撰写：哪些题型失败、失败原因归类、下一步只改哪一个变量。）")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="bm25_v1")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--idf", default="lucene", choices=["lucene", "okapi"])
    ap.add_argument("--field", default="index_text", choices=["index_text", "text"])
    ap.add_argument("--product-filter", action="store_true")
    ap.add_argument("--retriever", default="bm25", choices=["bm25", "dense", "hybrid", "rerank"])
    ap.add_argument("--rerank", default=None, help="data/processed/rerank 下的分数目录")
    ap.add_argument("--rerank-top", type=int, default=0, help="向量检索前多少名进入重排，0 = 全部")
    ap.add_argument("--emb", default="bge-m3_index_text", help="data/processed/emb 下的向量目录")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--w-bm25", type=float, default=1.0, help="加权 RRF 中 BM25 的权重，Dense 固定为 1")
    ap.add_argument("--tokenizer", default="precise", choices=["precise", "search"])
    ap.add_argument("--terms", default="insurance_terms.txt", help="src/resources 下的词典文件名")
    ap.add_argument("--golden", default=GOLDEN.name, help="dataset 下的评测集文件名")
    args = ap.parse_args()

    configure(RESOURCES / args.terms, mode=args.tokenizer)
    chunks = load_jsonl(CHUNKS)
    golden = load_jsonl(GOLDEN.parent / args.golden)
    if all(q.get("verified_by") == "learner" for q in golden):
        args.golden_status = f"学习者 {golden[0]['verified_at']} 核定"
    else:  # 未经学习者核定的题只能当参考，报告里必须写明
        args.golden_status = "AI 标注，待学习者核定"
    index = BM25([tokenize(c[args.field]) for c in chunks], k1=args.k1, b=args.b, idf_variant=args.idf)
    chunk_vecs = question_vecs = None
    if args.retriever != "bm25":
        emb = EMB_DIR / args.emb
        meta = json.loads((emb / "meta.json").read_text(encoding="utf-8"))
        # 向量和文本块、题目必须一一对应；切分或题目变了就要重跑 embed.py
        if meta["chunk_ids"] != [c["chunk_id"] for c in chunks] or meta["question_ids"] != [q["id"] for q in golden]:
            raise SystemExit(f"{emb} 与当前 chunks / golden 不一致，请重新运行 src/embed.py")
        chunk_vecs, question_vecs = np.load(emb / "chunks.npy"), np.load(emb / "questions.npy")
    rerank_scores = None
    if args.retriever == "rerank":
        rdir = ROOT / "data" / "processed" / "rerank" / args.rerank
        rmeta = json.loads((rdir / "meta.json").read_text(encoding="utf-8"))
        if rmeta["chunk_ids"] != [c["chunk_id"] for c in chunks] or rmeta["question_ids"] != [q["id"] for q in golden]:
            raise SystemExit(f"{rdir} 与当前 chunks / golden 不一致，请重新运行 src/rerank.py")
        rerank_scores = np.load(rdir / "scores.npy")
    retriever = Retriever(
        args.retriever, index, chunk_vecs, question_vecs, rrf_k=args.rrf_k, w_bm25=args.w_bm25,
        rerank_scores=rerank_scores, rerank_top=args.rerank_top,
    )
    results = evaluate(chunks, golden, retriever, args.k, product_filter=args.product_filter)

    (REPORTS / "runs").mkdir(parents=True, exist_ok=True)
    with (REPORTS / "runs" / f"{args.run_name}.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = write_report(results, args, len(chunks))

    s = summarize([r for r in results if r["type"] != "refuse"])
    print(f"{args.run_name}: Hit@{args.k}={s['hit']:.1%} Recall@{args.k}={s['recall']:.1%} MRR@10={s['mrr']:.3f}")
    print(f"report -> {report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
