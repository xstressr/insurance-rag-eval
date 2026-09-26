"""手写的检索 Agent：模型自己决定查什么、查几次、读哪条条款，最后给出带引用的答案。

先启动向量服务（Miniconda）：
    C:\\Users\\xstre\\miniconda3\\python.exe src/embed_server.py
再运行（uv）：
    uv run python src/agent.py --run-name agent_v1 --golden golden_v2.jsonl
    uv run python src/agent.py --run-name agent_v1_blind --golden golden_blind_v1.jsonl

循环：
    对话 = [系统提示, 问题]
    重复最多 max_steps 次：
        模型回复；没有工具调用 → 当作最终答案
        逐个执行工具调用（校验工具名和参数、拦截重复调用），把结果追加回对话
        调用了 final_answer → 结束
    步数用完 → 撤掉工具，要求模型根据已有信息作答

每个工具结果里的文本块都分配全局编号 C1、C2……，答案用这些编号引用。
输出格式与 answer.py 一致，所以 judge.py 可以直接评分。
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

from answer import LLM, PROMPTS, REFUSAL, parse_json, summarize
from citecheck import SEVERE, check
from evaluate import CHUNKS, EMB_DIR, GOLDEN, REPORTS, ROOT, SHORT, is_relevant, load_jsonl

EMBED_URL = "http://127.0.0.1:8765/embed"
PRODUCTS = {
    "太保阿基米德": "cpic_archimedes_2025",
    "泰康惠嘉保": "taikang_huijiabao_2026",
    "国寿康宁尊享": "chinalife_kangning_zunxiang_2024",
    "行业规范": "iac_ci_definitions_2020",
}
SEARCH_TOP = 5
SNIPPET_CHARS = 500
CLAUSE_CHARS = 6000

# 工具名要短：实测 DeepSeek 经网关调用时，search_clauses 会被写成 search_clause / search_cl
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "按语义检索保险条款，返回最相关的 5 条条款的片段（每条带编号、产品、条款号）。"
            "用户的口语和条款术语常常不同，查不到时换成条款可能使用的说法再查。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索语句，尽量用条款可能使用的术语"},
                    "products": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(PRODUCTS)},
                        "description": "只在这些产品中检索；不填则检索全部",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_clause",
            "description": "读取某一条款的完整原文。检索结果只是片段，需要完整规定（如例外、后半段）时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {"type": "string", "enum": list(PRODUCTS)},
                    "clause_id": {"type": "string", "description": "检索结果里显示的条款号，如 8.2、第十五条"},
                    "part": {"type": "string", "description": "国寿的“利益条款”或“基本条款”，其他产品不填"},
                },
                "required": ["product", "clause_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "给出最终答案并结束。",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "带 [Cn] 引用的回答"},
                    "answerable": {"type": "boolean", "description": "条款信息是否足以回答"},
                },
                "required": ["answer", "answerable"],
            },
        },
    },
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}

AGENT_VERSION = "a1"  # 循环版本；提示词版本见 AGENT_PROMPTS
# 回答规则沿用 p2 的 1～8 条，只把输出格式换成工具调用
_P2_RULES = PROMPTS["p2"].split("规则：", 1)[1].split("只输出一个 JSON", 1)[0].strip()
SYSTEM_PROMPT = f"""你是保险条款问答助手，可以使用工具检索条款。只能依据工具返回的条款内容回答，不能使用常识或记忆。

资料范围：太保阿基米德（2025）重大疾病保险、泰康惠嘉保2026重大疾病保险、国寿康宁尊享重大疾病保险三款产品的条款，以及《重大疾病保险的疾病定义使用规范（2020年修订版）》（行业规范）。

工作方法：
- 先用 search 检索。问题涉及多个产品时，对每个产品分别检索（用 products 参数限定）。
- 问题往往要多条条款才能答全（例如一般规定和例外、犹豫期内和犹豫期后、权利和限制）。答题前先想清楚还缺什么，再针对缺的部分检索。
- 检索结果只是片段。需要某条款的完整规定时，用 read_clause 读取全文。
- 查不到时，换成条款可能使用的术语再查，而不是马上放弃。
- 最多 {{max_steps}} 轮工具调用，信息足够就用 final_answer 作答。

回答规则（写进 final_answer）：
{_P2_RULES}
引用编号是工具结果里的 [Cn]。"""

# 提示词只增不改，旧版本保留以便复现。
# a2：agent_v1 的答案变长后开始夹带片段外的补充（3 题不忠实），且常不经 final_answer 直接输出文字。
AGENT_PROMPTS = {
    "a1": SYSTEM_PROMPT,
    "a2": SYSTEM_PROMPT
    + """

补充要求（优先于上面的规则）：
9. final_answer 中的每一句都必须带 [Cn] 引用。不要写片段原文没有的提醒、建议、推测或概括（例如“以保险单为准”“建议咨询保险公司”“一般来说”“其他产品也类似”），除非片段里写了。
10. 只回答问题问到的内容，不要罗列与问题无关的条款细节。
11. 必须调用 final_answer 提交答案，不要直接输出文字。""",
}


class Corpus:
    """Agent 工具背后的数据：文本块、向量、条款索引。"""

    def __init__(self) -> None:
        self.chunks = load_jsonl(CHUNKS)
        emb = EMB_DIR / "bge-m3_index_text"
        meta = json.loads((emb / "meta.json").read_text(encoding="utf-8"))
        assert meta["chunk_ids"] == [c["chunk_id"] for c in self.chunks], "向量与 chunks 不一致，请重跑 embed.py"
        self.vecs = np.load(emb / "chunks.npy")
        self.clauses: dict[tuple, list[int]] = {}
        for i, c in enumerate(self.chunks):
            if c["section"] != "guide":
                self.clauses.setdefault((c["doc_id"], c["part"], c["clause_id"]), []).append(i)

    def embed(self, text: str) -> np.ndarray:
        req = urllib.request.Request(
            EMBED_URL, data=json.dumps({"texts": [text]}).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return np.array(json.load(resp)["vectors"][0], dtype=np.float32)

    def where(self, c: dict) -> str:
        loc = "阅读指引" if c["section"] == "guide" else f"{c['part']} {c['clause_id']} {c['title']}".strip()
        return f"{SHORT[c['doc_id']]} | {loc}"


class Episode:
    """一道题的一次执行：维护引用编号、看过的文本块和轨迹。"""

    def __init__(self, corpus: Corpus, question: str = "", guard: int = 0) -> None:
        self.corpus = corpus
        self.question = question
        self.guard_left = guard  # 护栏还能退回几次
        self.guard_log: list[dict] = []
        self.blocks: list[dict] = []  # {"cid", "chunk_ids"}，与 answer.py 的 context 格式一致
        self.seen_calls: dict[str, str] = {}

    def block_meta(self) -> dict[str, tuple[str, set[str]]]:
        """{Cn: (片段全文, 所属文档)}，给引用校验用。"""
        by_id = {c["chunk_id"]: c for c in self.corpus.chunks}
        return {
            b["cid"]: ("".join(by_id[x]["text"] for x in b["chunk_ids"]), {by_id[x]["doc_id"] for x in b["chunk_ids"]})
            for b in self.blocks
        }

    def _new_block(self, idxs: list[int]) -> str:
        cid = f"C{len(self.blocks) + 1}"
        self.blocks.append({"cid": cid, "chunk_ids": [self.corpus.chunks[i]["chunk_id"] for i in idxs]})
        return cid

    def search(self, query: str, products: list[str] | None = None) -> str:
        docs = {PRODUCTS[p] for p in products} if products else None
        cos = self.corpus.vecs @ self.corpus.embed(query)
        hits, seen_clause = [], set()
        for i in np.argsort(-cos):
            c = self.corpus.chunks[i]
            if docs and c["doc_id"] not in docs:
                continue
            key = (c["doc_id"], c["part"], c["clause_id"], c["section"] == "guide")
            if key in seen_clause:  # 同一条款只出一次，要全文用 read_clause
                continue
            seen_clause.add(key)
            hits.append(int(i))
            if len(hits) == SEARCH_TOP:
                break
        lines = []
        for i in hits:
            c = self.corpus.chunks[i]
            cid = self._new_block([i])
            text = c["text"] if len(c["text"]) <= SNIPPET_CHARS else c["text"][:SNIPPET_CHARS] + "……（片段，全文用 read_clause）"
            lines.append(f"[{cid}] {self.corpus.where(c)} | 第{c['page_start']}页\n{text}")
        return "\n\n".join(lines) if lines else "没有检索到结果。"

    def read_clause(self, product: str, clause_id: str, part: str | None = None) -> str:
        doc = PRODUCTS[product]
        cands = [k for k in self.corpus.clauses if k[0] == doc and k[2] == clause_id.strip() and (not part or k[1] == part)]
        if not cands:
            return f"错误：{product} 没有条款号为“{clause_id}”的条款。请使用检索结果中显示的条款号。"
        if len(cands) > 1:
            return f"错误：{product} 的“{clause_id}”在多个部分都有（{'、'.join(k[1] for k in cands)}），请用 part 参数指定。"
        idxs = self.corpus.clauses[cands[0]]
        text = "".join(self.corpus.chunks[i]["text"] for i in idxs)
        if len(text) > CLAUSE_CHARS:
            text = text[:CLAUSE_CHARS] + f"……（全文 {len(text)} 字，已截断）"
        c0, c1 = self.corpus.chunks[idxs[0]], self.corpus.chunks[idxs[-1]]
        cid = self._new_block(idxs)
        return f"[{cid}] {self.corpus.where(c0)} | 第{c0['page_start']}-{c1['page_end']}页（全文）\n{text}"


def validate(name: str, raw_args: str) -> tuple[dict | None, str | None]:
    """程序层校验：工具名、JSON、必填字段、枚举值。返回 (参数, 错误信息)。"""
    if name not in TOOL_NAMES:
        return None, f"错误：没有名为“{name}”的工具。可用工具：{'、'.join(sorted(TOOL_NAMES))}。"
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        return None, "错误：参数不是合法的 JSON。"
    schema = next(t["function"]["parameters"] for t in TOOLS if t["function"]["name"] == name)
    if not isinstance(args, dict):
        return None, "错误：参数必须是 JSON 对象。"
    missing = [k for k in schema["required"] if k not in args]
    if missing:
        return None, f"错误：缺少必填参数 {missing}。"
    unknown = [k for k in args if k not in schema["properties"]]
    if unknown:  # 实测模型会把 products 写成 product
        return None, f"错误：未知参数 {unknown}，可用参数：{list(schema['properties'])}。"
    bad = []
    if name == "search":
        if not isinstance(args["query"], str) or not args["query"].strip():
            bad.append("query 必须是非空字符串")
        bad += [f"未知产品“{p}”" for p in args.get("products") or [] if p not in PRODUCTS]
    if name == "read_clause" and args["product"] not in PRODUCTS:
        bad.append(f"未知产品“{args['product']}”，可选：{'、'.join(PRODUCTS)}")
    if bad:
        return None, "错误：" + "；".join(bad) + "。"
    return args, None


FORCE_ANSWER = "工具调用次数已用完。请只根据已有检索结果，按回答规则直接给出答案，输出 JSON：{\"answerable\": true或false, \"answer\": \"...\"}"
ANSWER_ONLY = [t for t in TOOLS if t["function"]["name"] == "final_answer"]


def system_prompt(version: str, max_steps: int) -> str:
    return AGENT_PROMPTS[version].replace("{max_steps}", str(max_steps))


def guard_message(issues: list[dict]) -> str:
    lines = ["答案未通过引用校验，请修正后重新调用 final_answer："]
    for it in issues:
        num = "、".join(f"{n}{u}" for n, u in it["facts"])
        where = "、".join(it.get("cited", [])) or "无引用"
        hint = {
            "张冠李戴": "只出现在其他产品的片段里，请确认是否写错了产品",
            "上下文中不存在": "在所有检索到的片段里都找不到（包括自行推算的数字）",
            "未引用": "这段文字没有标注引用",
        }[it["kind"]]
        lines.append(f"- 【{it['kind']}】“{num}”（所引片段：{where}）{hint}。所在文字：…{it['sentence'][-60:]}")
    lines.append("修正要求：数字必须来自它所引用的片段；写错产品的请更正并引用正确的片段；片段里没有的数字请删除；没有引用的请补上引用。")
    return "\n".join(lines)


def execute_calls(
    ep: Episode, calls: list[dict], step: int, trajectory: list[dict], allow_guard: bool = True
) -> tuple[list[dict], dict | None]:
    """执行一轮里的全部工具调用：校验 → 去重 → 执行。返回 (工具结果消息, 最终答案或 None)。

    护栏：final_answer 提交时跑引用校验，有严重问题且还有退回次数，就把问题清单作为工具结果退回，
    本次提交不算数。强制作答轮（allow_guard=False）不启用，避免拿不到任何答案。"""
    tool_msgs, final = [], None
    for call in calls:
        name, raw = call["function"]["name"], call["function"]["arguments"]
        targs, err = validate(name, raw)
        sig = f"{name}:{json.dumps(targs, ensure_ascii=False, sort_keys=True)}" if targs else None
        if err:
            result = err
        elif name == "final_answer":
            issues = []
            if allow_guard and ep.guard_left > 0:
                issues = [i for i in check(targs["answer"], ep.block_meta(), ep.question) if i["kind"] in SEVERE]
            if issues:
                ep.guard_left -= 1
                ep.guard_log.append({"step": step, "answer": targs["answer"], "issues": issues})
                result = guard_message(issues)
            else:
                final = {"answer": targs["answer"], "answerable": targs["answerable"]}
                result = "已提交。"
        elif sig in ep.seen_calls:  # 幂等：同样的调用不重复执行，指向之前的结果
            result = f"重复调用：与之前的调用完全相同，结果见 {ep.seen_calls[sig]}。请换一个查询或直接作答。"
        else:
            first_cid = f"C{len(ep.blocks) + 1}"
            result = ep.search(**targs) if name == "search" else ep.read_clause(**targs)
            if not result.startswith("错误"):
                ep.seen_calls[sig] = f"{first_cid} 起的结果"
        trajectory.append({"step": step, "tool": name, "args": targs if targs else raw, "error": err, "result_head": result[:120]})
        tool_msgs.append({"role": "tool", "tool_call_id": call["id"], "content": result})
    return tool_msgs, final


def text_answer(msg: dict) -> dict:
    """模型没调用工具、直接输出文字时，当作最终答案解析。"""
    parsed = parse_json(msg["content"])
    return {"answer": parsed["answer"] if parsed else msg["content"], "answerable": parsed.get("answerable") if parsed else None}


def run_episode(q: dict, corpus: Corpus, llm: LLM, args: argparse.Namespace) -> dict:
    ep = Episode(corpus, q["question"], args.guard)
    messages = [
        {"role": "system", "content": system_prompt(args.agent_prompt, args.max_steps)},
        {"role": "user", "content": q["question"]},
    ]
    session = str(uuid.uuid5(uuid.NAMESPACE_URL, f"insurance-rag-eval/{args.run_name}/{q['id']}"))
    trajectory, usage = [], {"prompt": 0, "completion": 0, "reasoning": 0}
    final, stop, cached_all = None, "max_steps", True

    for step in range(1, args.max_steps + 2):
        final_round = step > args.max_steps
        if final_round:  # 步数用完：只留 final_answer 一个工具
            messages.append({"role": "user", "content": FORCE_ANSWER})
        resp = llm.chat_tools(messages, ANSWER_ONLY if final_round else TOOLS, session)
        cached_all &= resp["cached"]
        for k in usage:
            usage[k] += resp["usage"][k]
        msg = resp["message"]
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            final, stop = text_answer(msg), "text_answer"
            trajectory.append({"step": step, "tool": None, "note": "模型直接输出文字"})
            break
        tool_msgs, final = execute_calls(ep, calls, step, trajectory, allow_guard=not final_round)
        messages.extend(tool_msgs)
        if final:
            stop = "final_answer" if not final_round else "forced_answer"
            break
    return finalize(q, ep, final, stop, trajectory, usage, cached_all)


def uncited_sentences(answer: str) -> list[str]:
    """按句号、分号和换行切句，返回没有 [Cn] 引用的实质句子（忽略短于 8 字的标题或连接语）。"""
    parts = re.split(r"(?<=[。；！？])|\n", answer)
    return [p.strip() for p in parts if len(re.sub(r"[\s*#\-：:0-9.、（）()]", "", p)) >= 8 and not re.search(r"\[C\d+\]", p)]


def finalize(q: dict, ep: Episode, final: dict | None, stop: str, trajectory: list[dict], usage: dict, cached_all: bool) -> dict:
    """把一次执行整理成与 answer.py 一致的结果行，judge.py 可以直接评分。"""
    answer = (final or {}).get("answer") or ""
    answerable = (final or {}).get("answerable")
    chunks = ep.corpus.chunks
    by_id = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    cids = {b["cid"]: b for b in ep.blocks}
    cited = list(dict.fromkeys(re.findall(r"\[(C\d+)\]", answer)))
    relevant_cids = [
        b["cid"]
        for b in ep.blocks
        if any(is_relevant(chunks[by_id[x]], s) for x in b["chunk_ids"] for s in q["expected_sources"])
    ]
    tool_calls = [t for t in trajectory if t["tool"]]
    return {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "answer": answer,
        "refused": answerable is False or answer.strip().startswith(REFUSAL),
        "parse_ok": bool(answer),
        "finish_reason": stop,
        "context": ep.blocks,
        "context_chars": sum(chunks[by_id[x]]["n_chars"] for b in ep.blocks for x in b["chunk_ids"]),
        "relevant_in_context": relevant_cids,
        "cited": cited,
        "invalid_citations": [c for c in cited if c not in cids],
        "cited_relevant": [c for c in cited if c in relevant_cids],
        "must_include": q["must_include"],
        "must_not_include_hits": [s for s in q.get("must_not_include", []) if s in answer],
        "usage": usage,
        "seconds": None,
        "cached": cached_all,
        "stop": stop,
        "steps": len({t["step"] for t in trajectory}),
        "n_tool_calls": len(tool_calls),
        "n_tool_errors": sum(bool(t.get("error")) for t in tool_calls),
        "n_duplicate_calls": sum(str(t.get("result_head", "")).startswith("重复调用") for t in tool_calls),
        "tools_used": [t["tool"] for t in tool_calls],
        "answer_chars": len(answer),
        "guard_rejections": len(ep.guard_log),
        "guard_log": ep.guard_log,
        "residual_severe": [i for i in check(answer, ep.block_meta(), q["question"]) if i["kind"] in SEVERE],
        "uncited_sentences": uncited_sentences(answer),
        "trajectory": trajectory,
    }


def write_report(rows: list[dict], s: dict, args: argparse.Namespace, llm: LLM) -> Path:
    n = len(rows)
    calls = sum(r["n_tool_calls"] for r in rows)
    lines = [
        f"# Agent 报告 · {args.run_name}",
        "",
        f"> 自动生成（{date.today()}）。评测集 `dataset/{args.golden}`。{getattr(args, 'engine', '手写循环')} `{AGENT_VERSION}`，提示词 `{args.agent_prompt}`，模型 `{llm.model}`，最多 {args.max_steps} 轮工具调用，护栏退回上限 {getattr(args, 'guard', 0)} 次。",
        "",
        "## 指标",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 上下文命中率（看过的块里有证据） | {s['context_hit']:.1%} |",
        f"| 误拒率 / 正确拒答率 | {s['false_refusal']:.1%} / {'-' if s['correct_refusal'] is None else format(s['correct_refusal'], '.1%')} |",
        f"| 平均轮数 / 平均工具调用 | {sum(r['steps'] for r in rows) / n:.1f} / {calls / n:.1f} |",
        f"| 工具调用出错（校验拦截） | {sum(r['n_tool_errors'] for r in rows)} / {calls} |",
        f"| 重复调用（被拦截） | {sum(r['n_duplicate_calls'] for r in rows)} |",
        f"| 结束方式 | {', '.join(f'{k} {v}' for k, v in sorted(__import__('collections').Counter(r['stop'] for r in rows).items()))} |",
        f"| 用过 read_clause 的题 | {sum('read_clause' in r['tools_used'] for r in rows)} |",
        f"| 平均看过的字数 | {s['avg_context_chars']:.0f} |",
        f"| 平均答案字数 | {sum(r['answer_chars'] for r in rows) / n:.0f} |",
        f"| 护栏退回 | {sum(r.get('guard_rejections', 0) for r in rows)} 次（{sum(bool(r.get('guard_rejections')) for r in rows)} 题）；最终答案仍有严重问题 {sum(bool(r.get('residual_severe')) for r in rows)} 题 |",
        f"| 没带引用的句子 | {sum(len(r['uncited_sentences']) for r in rows)}（涉及 {sum(bool(r['uncited_sentences']) for r in rows)} 题） |",
        f"| token（输入 / 输出 / 其中思考） | {s['tokens_prompt']} / {s['tokens_completion']} / {s['tokens_reasoning']} |",
        "",
        "## 逐题轨迹",
        "",
        "| 题 | 类型 | 轮数 | 工具调用 | 证据 | 拒答 | 答案 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        tr = " → ".join(
            f"{t['tool']}({(t['args'].get('query') or t['args'].get('clause_id', '')) if isinstance(t['args'], dict) else '?'}{'✗' if t.get('error') else ''})"
            for t in r["trajectory"]
            if t["tool"]
        ).replace("|", "／")
        ans = r["answer"].replace("\n", " ").replace("|", "／")
        lines.append(
            f"| {r['id']} | {r['type']} | {r['steps']} | {tr} | {'、'.join(r['relevant_in_context']) or '无'} | {'是' if r['refused'] else ''} | {ans} |"
        )
    path = REPORTS / f"{args.run_name}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--golden", default=GOLDEN.name)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--agent-prompt", default="a2", help="AGENT_PROMPTS 中的版本")
    ap.add_argument("--guard", type=int, default=1, help="引用校验护栏最多退回几次，0 = 关闭")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="逗号分隔的题号，只跑这些题")
    args = ap.parse_args()

    corpus = Corpus()
    golden = load_jsonl(GOLDEN.parent / args.golden)
    if args.only:
        keep = set(args.only.split(","))
        golden = [q for q in golden if q["id"] in keep]
    golden = golden[: args.limit]
    llm = LLM(None)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda q: run_episode(q, corpus, llm, args), golden))

    (REPORTS / "runs").mkdir(parents=True, exist_ok=True)
    with (REPORTS / "runs" / f"{args.run_name}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = summarize(rows)
    report = write_report(rows, s, args, llm)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, ensure_ascii=False))
    print(f"report -> {report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
