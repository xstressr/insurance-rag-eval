"""检索 → 组装上下文 → 大模型生成带引用的答案 → 程序检查引用与拒答。

用法：
    uv run python src/answer.py --run-name gen_v1
    uv run python src/answer.py --run-name gen_v1_blind --golden golden_blind_v1.jsonl --emb bge-m3_index_text_blind

模型配置读项目根目录的 .env（LLM_BASE_URL / LLM_API_KEY / LLM_MODEL），key 不打印、不落盘。
同一个模型 + 同一份提示词的响应缓存在 data/processed/llm_cache/，重跑评测不重复花钱。

这一步只做程序能判断的指标：检索是否把证据送进了上下文、是否拒答、引用是否真实、是否指向正确证据。
答案“对不对、有没有编造”要靠 LLM 评委，放在下一步。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from bm25 import BM25, RESOURCES, configure, tokenize
from evaluate import CHUNKS, EMB_DIR, GOLDEN, REPORTS, ROOT, SHORT, Retriever, detect_products, is_relevant, load_jsonl

CACHE = ROOT / "data" / "processed" / "llm_cache"
USER_AGENT = "insurance-rag-eval/0.1"
REFUSAL = "根据提供的条款无法回答"

PRODUCT_NAME = {
    "cpic_archimedes_2025": "太保阿基米德（2025）重大疾病保险",
    "taikang_huijiabao_2026": "泰康惠嘉保2026重大疾病保险",
    "chinalife_kangning_zunxiang_2024": "国寿康宁尊享重大疾病保险",
    "iac_ci_definitions_2020": "重大疾病保险的疾病定义使用规范（2020年修订版）",
}

# 改动提示词要同步改版本号：报告和缓存都靠它区分结果来自哪一版提示词。
PROMPT_VERSION = "p1"
SYSTEM_PROMPT = f"""你是保险条款问答助手。只能依据用户给出的【条款片段】回答，不能使用片段以外的常识或记忆。

规则：
1. 每一句陈述事实的话，句末用方括号标注依据的片段编号，例如 [C1] 或 [C1][C3]。
2. 片段不足以回答时，answer 以“{REFUSAL}”开头，再用一句话说明缺少什么信息；answerable 设为 false。
3. 片段只能回答一部分时，回答能回答的部分并注明，另一部分说明条款未提及；answerable 设为 true。
4. 对比题要分别说明每个产品，再给出比较结论。
5. 回答简洁，使用条款原文中的数字和术语，不要改写数字。

只输出一个 JSON 对象，不要输出其他内容：
{{"answerable": true 或 false, "answer": "带 [Cn] 引用的回答"}}"""


def build_context(chunks: list[dict], order: list[int], k: int, char_budget: int) -> list[dict]:
    """按检索名次取前 k 块：去掉正文重复的块，超出字数预算就停止。编号 C1..Ck 与名次一致。"""
    ctx, seen, used = [], set(), 0
    for i in order:
        c = chunks[i]
        if c["text"] in seen:
            continue
        if used + c["n_chars"] > char_budget and ctx:
            break
        seen.add(c["text"])
        used += c["n_chars"]
        ctx.append({"cid": f"C{len(ctx) + 1}", "idx": i, "chunk_id": c["chunk_id"]})
        if len(ctx) == k:
            break
    return ctx


def render_context(chunks: list[dict], ctx: list[dict]) -> str:
    parts = []
    for item in ctx:
        c = chunks[item["idx"]]
        where = "阅读指引" if c["section"] == "guide" else f"{c['part']} {c['clause_id']} {c['title']}".strip()
        pages = f"第{c['page_start']}页" if c["page_start"] == c["page_end"] else f"第{c['page_start']}-{c['page_end']}页"
        parts.append(f"[{item['cid']}] {PRODUCT_NAME[c['doc_id']]} | {where} | {pages}\n{c['text']}")
    return "\n\n".join(parts)


def parse_json(text: str) -> dict | None:
    """模型偶尔会在 JSON 外面包代码块或多说一句，取第一个 {...}。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) and "answer" in obj else None


class LLM:
    def __init__(self, max_tokens: int) -> None:
        load_dotenv(ROOT / ".env")
        self.model = os.environ["LLM_MODEL"]
        self.max_tokens = max_tokens
        self.client = OpenAI(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
            default_headers={"User-Agent": USER_AGENT},
            timeout=180,
            max_retries=2,
        )

    def chat(self, system: str, user: str, session: str) -> dict:
        """返回 {content, usage, model, seconds, cached}。缓存键 = 模型 + 提示词全文。"""
        key = hashlib.sha256(json.dumps([self.model, system, user], ensure_ascii=False).encode()).hexdigest()[:24]
        path = CACHE / f"{key}.json"
        if path.exists():
            return {**json.loads(path.read_text(encoding="utf-8")), "cached": True}
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=self.max_tokens,
            # OpenCode Go 要求：每个对话一个稳定的会话 ID，用于路由和提示词缓存
            extra_headers={"x-opencode-session": session},
        )
        u = resp.usage
        details = getattr(u, "completion_tokens_details", None)
        out = {
            "content": resp.choices[0].message.content or "",
            "finish_reason": resp.choices[0].finish_reason,
            "model": resp.model,
            "usage": {
                "prompt": u.prompt_tokens,
                "completion": u.completion_tokens,
                "reasoning": getattr(details, "reasoning_tokens", None) or 0,
            },
            "seconds": round(time.time() - t0, 1),
        }
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return {**out, "cached": False}


def run_one(q: dict, qi: int, chunks: list[dict], retriever: Retriever, llm: LLM, args: argparse.Namespace) -> dict:
    order, _ = retriever.rank(qi, tokenize(q["question"]))
    products = detect_products(q["question"])
    if products:
        order = [i for i in order if chunks[i]["doc_id"] in products]
    ctx = build_context(chunks, order, args.k, args.char_budget)
    user = f"【条款片段】\n{render_context(chunks, ctx)}\n\n【问题】\n{q['question']}"
    session = str(uuid.uuid5(uuid.NAMESPACE_URL, f"insurance-rag-eval/{args.run_name}/{q['id']}"))
    resp = llm.chat(SYSTEM_PROMPT, user, session)

    parsed = parse_json(resp["content"])
    answer = parsed["answer"] if parsed else resp["content"]
    refused = (parsed is not None and parsed.get("answerable") is False) or answer.strip().startswith(REFUSAL)

    cids = {item["cid"]: item for item in ctx}
    cited = list(dict.fromkeys(re.findall(r"\[(C\d+)\]", answer)))
    relevant_cids = [
        item["cid"] for item in ctx if any(is_relevant(chunks[item["idx"]], s) for s in q["expected_sources"])
    ]
    return {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "answer": answer,
        "refused": refused,
        "parse_ok": parsed is not None,
        "finish_reason": resp.get("finish_reason"),
        "context": [{"cid": it["cid"], "chunk_id": it["chunk_id"]} for it in ctx],
        "relevant_in_context": relevant_cids,
        "cited": cited,
        "invalid_citations": [c for c in cited if c not in cids],
        "cited_relevant": [c for c in cited if c in relevant_cids],
        "must_include": q["must_include"],
        "must_not_include_hits": [s for s in q.get("must_not_include", []) if s in answer],
        "usage": resp["usage"],
        "seconds": resp["seconds"],
        "cached": resp["cached"],
    }


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["type"] != "refuse"]
    ref = [r for r in rows if r["type"] == "refuse"]
    ctx_ok = [r for r in ans if r["relevant_in_context"]]
    cited_rows = [r for r in ans if r["cited"]]
    n_cited = sum(len(r["cited"]) for r in rows)
    return {
        "n_answerable": len(ans),
        "n_refuse": len(ref),
        "context_hit": len(ctx_ok) / len(ans),
        "false_refusal": sum(r["refused"] for r in ans) / len(ans),
        "false_refusal_ctx_ok": sum(r["refused"] for r in ctx_ok) / len(ctx_ok) if ctx_ok else 0.0,
        "correct_refusal": sum(r["refused"] for r in ref) / len(ref) if ref else None,
        "answer_has_citation": len(cited_rows) / len(ans),
        "invalid_citation_rate": sum(len(r["invalid_citations"]) for r in rows) / n_cited if n_cited else 0.0,
        # 上下文里有证据、模型也作答了：它引用的块里有没有真正的证据
        "cites_evidence": (
            sum(bool(r["cited_relevant"]) for r in ctx_ok if not r["refused"])
            / max(1, sum(not r["refused"] for r in ctx_ok))
        ),
        "parse_fail": sum(not r["parse_ok"] for r in rows),
        "truncated": sum(r["finish_reason"] == "length" for r in rows),
        "tokens_prompt": sum(r["usage"]["prompt"] for r in rows),
        "tokens_completion": sum(r["usage"]["completion"] for r in rows),
        "tokens_reasoning": sum(r["usage"]["reasoning"] for r in rows),
        "api_calls": sum(not r["cached"] for r in rows),
    }


def write_report(rows: list[dict], s: dict, args: argparse.Namespace, llm: LLM, n_chunks: int) -> Path:
    pct = lambda x: f"{x:.1%}"  # noqa: E731
    lines = [
        f"# 生成评测报告 · {args.run_name}",
        "",
        f"> 自动生成（{date.today()}）。评测集 `dataset/{args.golden}`。",
        "",
        "## 版本",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 模型 | `{llm.model}`（OpenCode Go，temperature=0，max_tokens={args.max_tokens}） |",
        f"| 提示词 | `{PROMPT_VERSION}`（`src/answer.py` 的 SYSTEM_PROMPT） |",
        f"| 检索 | 向量检索 bge-m3（`{args.emb}`）+ 产品过滤，文本块 {n_chunks} 个 |",
        f"| 上下文 | 前 {args.k} 块，去重，字数预算 {args.char_budget} |",
        "",
        "## 指标",
        "",
        "| 指标 | 值 | 含义 |",
        "|---|---|---|",
        f"| 上下文命中率 | {pct(s['context_hit'])} | 可回答题中，上下文里至少有一块证据（等于检索的 Hit@{args.k}） |",
        f"| 误拒率 | {pct(s['false_refusal'])} | 可回答题被拒答 |",
        f"| 误拒率（证据已在上下文） | {pct(s['false_refusal_ctx_ok'])} | 证据明明给了还拒答，属于生成问题 |",
        f"| 正确拒答率 | {pct(s['correct_refusal']) if s['correct_refusal'] is not None else '-'} | 拒答题被正确拒答 |",
        f"| 答案带引用 | {pct(s['answer_has_citation'])} | 可回答题的答案里至少有一个 [Cn] |",
        f"| 无效引用率 | {pct(s['invalid_citation_rate'])} | 引用了上下文里不存在的编号 |",
        f"| 引用到证据 | {pct(s['cites_evidence'])} | 证据在上下文且模型作答时，引用里包含证据块 |",
        f"| JSON 解析失败 / 被截断 | {s['parse_fail']} / {s['truncated']} | |",
        f"| token（输入 / 输出 / 其中思考） | {s['tokens_prompt']} / {s['tokens_completion']} / {s['tokens_reasoning']} | 本次实际调用 {s['api_calls']} 次，其余来自缓存 |",
        "",
        "## 逐题",
        "",
        "| 题 | 类型 | 证据在上下文 | 拒答 | 引用 | 引用到证据 | 答案 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ans = r["answer"].replace("\n", " ").replace("|", "／")
        lines.append(
            f"| {r['id']} | {r['type']} | {'、'.join(r['relevant_in_context']) or '无'} | {'是' if r['refused'] else ''} | "
            f"{''.join(r['cited'])} | {'、'.join(r['cited_relevant'])} | {ans} |"
        )
    path = REPORTS / f"{args.run_name}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--golden", default=GOLDEN.name)
    ap.add_argument("--emb", default="bge-m3_index_text")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--char-budget", type=int, default=4000)
    ap.add_argument("--max-tokens", type=int, default=4096)  # 推理模型的思考也占这个额度
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 题，用于冒烟测试")
    args = ap.parse_args()

    configure(RESOURCES / "insurance_terms_v2.txt", mode="search")
    chunks = load_jsonl(CHUNKS)
    golden = load_jsonl(GOLDEN.parent / args.golden)
    emb = EMB_DIR / args.emb
    meta = json.loads((emb / "meta.json").read_text(encoding="utf-8"))
    if meta["chunk_ids"] != [c["chunk_id"] for c in chunks] or meta["question_ids"] != [q["id"] for q in golden]:
        raise SystemExit(f"{emb} 与当前 chunks / golden 不一致，请重新运行 src/embed.py")
    index = BM25([tokenize(c["index_text"]) for c in chunks])  # dense 模式下不参与排序，Retriever 需要它
    retriever = Retriever("dense", index, np.load(emb / "chunks.npy"), np.load(emb / "questions.npy"))
    llm = LLM(args.max_tokens)

    todo = list(enumerate(golden))[: args.limit]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda p: run_one(p[1], p[0], chunks, retriever, llm, args), todo))

    (REPORTS / "runs").mkdir(parents=True, exist_ok=True)
    with (REPORTS / "runs" / f"{args.run_name}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = summarize(rows)
    report = write_report(rows, s, args, llm, len(chunks))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, ensure_ascii=False))
    print(f"report -> {report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
