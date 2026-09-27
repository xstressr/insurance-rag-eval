"""检索 → 组装上下文 → 大模型生成带引用的答案 → 程序检查引用与拒答。

用法：
    uv run python src/answer.py --run-name gen_v3_p2 --golden golden_v2.jsonl   # 默认：expand=1、预算 8000、提示词 p2
    uv run python src/answer.py --run-name gen_v1 --expand 0 --char-budget 4000 --prompt p1   # 复现 gen_v1
    uv run python src/answer.py --run-name gen_v3_p2_blind --golden golden_blind_v1.jsonl --emb bge-m3_index_text_blind
    uv run python src/answer.py --run-name gen_v4_d2_blind5 --golden golden_blind_v5.jsonl --emb bge-m3_index_text_blind5 --decompose d2 --sub-k 5
      # 拆子问题（多查询检索），需要 embed_server 在跑；协议见 reports/decompose_v1_protocol.md

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
import urllib.request
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
EMBED_URL = os.environ.get("EMBED_URL", "http://127.0.0.1:8765/embed")

PRODUCT_NAME = {
    "cpic_archimedes_2025": "太保阿基米德（2025）重大疾病保险",
    "taikang_huijiabao_2026": "泰康惠嘉保2026重大疾病保险",
    "chinalife_kangning_zunxiang_2024": "国寿康宁尊享重大疾病保险",
    "iac_ci_definitions_2020": "重大疾病保险的疾病定义使用规范（2020年修订版）",
}

# 提示词只增不改：新版本另起一个键，旧版本保留，报告和缓存都能区分结果来自哪一版。
_OUTPUT = """只输出一个 JSON 对象，不要输出其他内容：
{"answerable": true 或 false, "answer": "带 [Cn] 引用的回答"}"""

PROMPTS = {
    "p1": f"""你是保险条款问答助手。只能依据用户给出的【条款片段】回答，不能使用片段以外的常识或记忆。

规则：
1. 每一句陈述事实的话，句末用方括号标注依据的片段编号，例如 [C1] 或 [C1][C3]。
2. 片段不足以回答时，answer 以“{REFUSAL}”开头，再用一句话说明缺少什么信息；answerable 设为 false。
3. 片段只能回答一部分时，回答能回答的部分并注明，另一部分说明条款未提及；answerable 设为 true。
4. 对比题要分别说明每个产品，再给出比较结论。
5. 回答简洁，使用条款原文中的数字和术语，不要改写数字。

{_OUTPUT}""",
    # p2 在 p1 基础上加 6～8 三条通用原则，针对 gen_v1 / ctx 实验中的三类生成失败：
    # 拘泥字面（用户用词和条款术语不同就拒答）、不敢从条款范围推断、把举例当成一般规则。
    "p2": f"""你是保险条款问答助手。只能依据用户给出的【条款片段】回答，不能使用片段以外的常识或记忆。

规则：
1. 每一句陈述事实的话，句末用方括号标注依据的片段编号，例如 [C1] 或 [C1][C3]。
2. 片段不足以回答时，answer 以“{REFUSAL}”开头，再用一句话说明缺少什么信息；answerable 设为 false。
3. 片段只能回答一部分时，回答能回答的部分并注明，另一部分说明条款未提及；answerable 设为 true。
4. 对比题要分别说明每个产品，再给出比较结论；某个产品在片段中没有相关内容时，明确指出。
5. 回答简洁，使用条款原文中的数字和术语，不要改写数字。
6. 按含义而不是字面匹配：用户的说法常与条款术语不同（口语、俗称、行业通称）。只要片段里的约定在含义上回答了问题，就据此回答，并说明条款中的原文说法；不要因为片段没有出现用户用的那个词就拒答。
7. 可以依据条款的适用范围作推断：条款明确规定某项权益或责任只适用于特定情形时，可以据此说明其他情形不适用，并注明“条款仅约定了……”作为依据。不要推断片段没有涉及的事实。
8. 区分规则与举例：片段中的举例、示例、演示数据只用于说明，不代表一般规则或具体费率。问题问的是一般性规则或数额、而片段只有举例时，说明条款未提供，可提及举例但必须注明仅为示例。

{_OUTPUT}""",
}


# 拆子问题（多查询检索）：一次不带思考的调用把问题拆成检索用的子问题。
# 只拆出 1 个时，检索和生成与不拆完全相同（生成调用命中同一份缓存）。
DECOMPOSE_PROMPTS = {
    "d1": """你为保险条款检索系统拆分用户问题。用户的一句话里常常连问好几件事，一次检索只能找到其中一件的条款。

规则：
1. 问题只问一件事时，只输出一个子问题，照抄原问题。
2. 问题问了几件需要分别查条款的事，就拆成几个子问题，最多 4 个。
3. 每个子问题要能单独拿去检索：带上产品名（原问题提到的话）和主语，尽量用条款里的说法（如“等待期”“宽限期”“现金价值”“受益人”“保单贷款”“责任免除”）。
4. 对比几款产品的同一件事，按产品各拆一个子问题。
5. 不要添加原问题没有问的事。

只输出 JSON：{"subquestions": ["子问题1", "子问题2"]}""",
    # d2：d1 拆出的子问题仍是口语，检索不到第二条条款（开发集 v3、v5 的多证据题）。
    # 改成像 Agent 那样写关键词式的检索词，并允许补查答案依赖的相关条款。
    "d2": """你为保险条款检索系统写检索词。条款按条组织，常见条款有：保险责任、责任免除、等待期、疾病定义（重度 / 中度 / 轻度）、
保险期间、投保年龄、保险费的交纳与宽限期、合同效力中止与恢复、犹豫期、解除合同与现金价值、保单贷款、受益人、
保险事故通知、保险金申请所需证明和资料、保险金的申请与给付、未还款项、争议处理、合同内容变更。

规则：
1. 找出回答问题需要查的每一条条款，每条写一个检索词，最多 4 个。只问一件事、一条条款就能回答时，只写 1 个。
2. 检索词写成关键词，用条款里的说法，不用口语。例如“能保到多大岁数”写成“保险期间 保障期限”，“贷款后生病赔付有没有影响”写成“保险金给付 扣除 未还贷款本息”。
3. 答案还依赖别的条款时也要查：问能不能赔某种病，要查这种病的疾病定义，并查有没有较轻一档（中度 / 轻度）；问理赔或给付，要查申请与给付条款。
4. 对比几款产品时，每款产品各写一个检索词，并写上产品名。
5. 原问题提到产品名时，每个检索词都带上产品名。

只输出 JSON：{"subquestions": ["检索词1", "检索词2"]}""",
}
NO_THINKING = {"thinking": {"type": "disabled"}}
# 至少拆出几个子问题才走多查询检索。d1 拆出 1 个时照抄原问题，不必再检索；d2 的 1 个检索词也是改写过的，要用。
SPLIT_MIN = {"d1": 2, "d2": 1}


def parse_subquestions(text: str) -> list[str]:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        subs = json.loads(m.group(0))["subquestions"] if m else []
    except (json.JSONDecodeError, KeyError, TypeError):
        subs = []
    if not isinstance(subs, list):  # 模型偶尔给一个字符串，别把它按字拆开
        return []
    return [s.strip() for s in subs if isinstance(s, str) and s.strip()][:4]


def merge_seeds(orders: list[list[int]], ks: list[int]) -> list[int]:
    """轮流从每个排序里取下一名，去重；第 i 个排序最多取 ks[i] 个。原问题的排序放第一个。"""
    seeds, seen = [], set()
    for r in range(max(ks)):
        for order, k in zip(orders, ks):
            if r < k and r < len(order) and order[r] not in seen:
                seen.add(order[r])
                seeds.append(order[r])
    return seeds


def build_context(
    chunks: list[dict], order: list[int], k: int, char_budget: int, expand: int = 0, whole_clause_max: int = 0
) -> list[dict]:
    """按检索名次取前 k 个“种子块”组装上下文，编号 C1..Cn 与名次一致。

    expand=0：每个种子块单独成段（gen_v1 的做法），去掉正文重复的块。
    expand=N：small-to-big。条款过长会被切成多块，命中的那一块不一定含答案，
      所以把同一条款里种子块前后各 N 块一起带上，同一条款的块合并成一段、按原文顺序排列。
    whole_clause_max>0（需 expand>0）：整条条款不超过这么多字时，带上整条条款，而不只是前后 N 块。
    超出字数预算就停止（第一段总会放进去）。
    """
    siblings: dict[tuple, dict[int, int]] = {}
    for i, c in enumerate(chunks):
        siblings.setdefault((c["doc_id"], c["part"], c["clause_id"]), {})[c["sub_index"]] = i

    blocks: list[dict] = []
    by_clause: dict[tuple, dict] = {}
    seen_text, used = set(), 0
    for i in order[:k]:
        c = chunks[i]
        key = (c["doc_id"], c["part"], c["clause_id"])
        lo, hi = c["sub_index"] - expand, c["sub_index"] + expand
        if expand and whole_clause_max and sum(chunks[j]["n_chars"] for j in siblings[key].values()) <= whole_clause_max:
            lo, hi = min(siblings[key]), max(siblings[key])  # 条款不长就整条带上，和 Agent 的 read_clause 一样
        want = [siblings[key][j] for j in range(lo, hi + 1) if j in siblings[key]]
        block = by_clause.get(key) if expand else None
        new = [j for j in want if chunks[j]["text"] not in seen_text and (block is None or j not in block["idxs"])]
        if not new:
            continue
        cost = sum(chunks[j]["n_chars"] for j in new)
        if used + cost > char_budget and blocks:
            break
        used += cost
        seen_text.update(chunks[j]["text"] for j in new)
        if block is None:
            block = {"cid": f"C{len(blocks) + 1}", "idxs": []}
            blocks.append(block)
            if expand:
                by_clause[key] = block
        block["idxs"] = sorted(set(block["idxs"]) | set(new), key=lambda j: chunks[j]["sub_index"])
    for b in blocks:
        b["chunk_ids"] = [chunks[j]["chunk_id"] for j in b["idxs"]]
    return blocks


def select_seeds(chunks: list[dict], order: list[int], products: set[str], k: int, quota: int = 0, neighbors: int = 0) -> list[int]:
    """从检索排序中挑“种子块”，交给 build_context 组装。

    quota>0：问题提到两个及以上产品（对比题）时，每个产品各取前 quota 块，再按原名次合并，
      避免上下文被排名靠前的单一产品占满。
    neighbors>0：对前 neighbors 个种子块，把同一文档、同一部分里前后相邻的条款（按原文顺序）
      也作为种子追加在最后，优先级最低，字数预算不够时先被舍弃。针对“完整答案要跨相邻条款”。
    """
    if quota and len(products) >= 2:
        picked = set()
        for doc in products:
            picked.update([i for i in order if chunks[i]["doc_id"] == doc][:quota])
        seeds = [i for i in order if i in picked]
    else:
        seeds = order[:k]
    if neighbors:
        clause_seq: dict[tuple, list[tuple]] = {}
        first_chunk: dict[tuple, int] = {}
        for i, c in enumerate(chunks):
            if c["section"] == "guide":
                continue
            key = (c["doc_id"], c["part"], c["clause_id"])
            if key not in first_chunk:
                first_chunk[key] = i
                clause_seq.setdefault((c["doc_id"], c["part"]), []).append(key)
        extra = []
        for i in seeds[:neighbors]:
            c = chunks[i]
            if c["section"] == "guide":
                continue
            seq = clause_seq[(c["doc_id"], c["part"])]
            pos = seq.index((c["doc_id"], c["part"], c["clause_id"]))
            for j in (pos - 1, pos + 1):
                if 0 <= j < len(seq) and first_chunk[seq[j]] not in seeds + extra:
                    extra.append(first_chunk[seq[j]])
        seeds = seeds + extra
    return seeds


def render_context(chunks: list[dict], ctx: list[dict]) -> str:
    parts = []
    for item in ctx:
        first, last = chunks[item["idxs"][0]], chunks[item["idxs"][-1]]
        where = "阅读指引" if first["section"] == "guide" else f"{first['part']} {first['clause_id']} {first['title']}".strip()
        p0, p1 = first["page_start"], last["page_end"]
        pages = f"第{p0}页" if p0 == p1 else f"第{p0}-{p1}页"
        text = "".join(chunks[j]["text"] for j in item["idxs"])
        parts.append(f"[{item['cid']}] {PRODUCT_NAME[first['doc_id']]} | {where} | {pages}\n{text}")
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
    def __init__(
        self, max_tokens: int | None, model: str | None = None, prefix: str = "LLM", extra_body: dict | None = None
    ) -> None:
        """prefix 决定读哪组环境变量：LLM_*（生成，默认 OpenCode Go）或 JUDGE_*（评委，可换服务商）。"""
        load_dotenv(ROOT / ".env")
        self.prefix = prefix
        self.extra_body = extra_body or {}  # 服务商特有参数，例如 GLM 的 thinking / reasoning_effort
        self.model = model or os.environ[f"{prefix}_MODEL"]
        self.max_tokens = max_tokens
        self.client = OpenAI(
            base_url=os.environ[f"{prefix}_BASE_URL"],
            api_key=os.environ[f"{prefix}_API_KEY"],
            default_headers={"User-Agent": USER_AGENT},
            timeout=180,
            max_retries=2,
        )

    def chat(self, system: str, user: str, session: str) -> dict:
        """返回 {content, usage, model, seconds, cached}。缓存键 = 模型 + 提示词全文。"""
        # 默认服务商沿用旧的缓存键（保证旧结果可复现）；其他服务商加上前缀，避免同名模型串用缓存
        parts = [self.model, system, user] if self.prefix == "LLM" else [self.prefix, self.model, system, user]
        if self.extra_body:
            parts.append(self.extra_body)
        key = hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:24]
        path = CACHE / f"{key}.json"
        if path.exists():
            return {**json.loads(path.read_text(encoding="utf-8")), "cached": True}
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            # None = 不传，用模型自己的最大输出上限（推理模型的思考也占这个额度）
            **({"max_tokens": self.max_tokens} if self.max_tokens else {}),
            # OpenCode Go 要求：每个对话一个稳定的会话 ID，用于路由和提示词缓存
            extra_headers={"x-opencode-session": session},
            extra_body=self.extra_body or None,
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
        # 被截断的响应不缓存，否则重跑会一直读到这个坏结果
        if out["finish_reason"] != "length":
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return {**out, "cached": False}

    def chat_tools(self, messages: list[dict], tools: list[dict], session: str) -> dict:
        """多轮 + 工具调用。缓存键 = 模型 + 完整对话历史 + 工具定义，所以整条轨迹可以原样回放。

        返回 {message, finish_reason, model, usage, seconds, cached}，message 可以直接追加回对话历史。
        """
        key = hashlib.sha256(
            json.dumps(["tools", self.prefix, self.model, messages, tools], ensure_ascii=False).encode()
        ).hexdigest()[:24]
        path = CACHE / f"{key}.json"
        if path.exists():
            return {**json.loads(path.read_text(encoding="utf-8")), "cached": True}
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=0,
            **({"max_tokens": self.max_tokens} if self.max_tokens else {}),
            extra_headers={"x-opencode-session": session},
            extra_body=self.extra_body or None,
        )
        choice = resp.choices[0]
        msg = {"role": "assistant", "content": choice.message.content or ""}
        if choice.message.tool_calls:
            msg["tool_calls"] = [
                {"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}}
                for t in choice.message.tool_calls
            ]
        # DeepSeek 思考模式下，同一轮工具调用中要把 reasoning_content 原样带回
        reasoning = getattr(choice.message, "reasoning_content", None)
        if reasoning:
            msg["reasoning_content"] = reasoning
        u = resp.usage
        details = getattr(u, "completion_tokens_details", None)
        out = {
            "message": msg,
            "finish_reason": choice.finish_reason,
            "model": resp.model,
            "usage": {
                "prompt": u.prompt_tokens,
                "completion": u.completion_tokens,
                "reasoning": getattr(details, "reasoning_tokens", None) or 0,
            },
            "seconds": round(time.time() - t0, 1),
        }
        if out["finish_reason"] != "length":
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return {**out, "cached": False}


def embed_query(text: str) -> np.ndarray:
    """子问题的向量现算（原问题的向量是预先算好的）。需要 embed_server 在跑。"""
    req = urllib.request.Request(
        EMBED_URL, data=json.dumps({"texts": [text]}).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return np.array(json.load(resp)["vectors"][0], dtype=np.float32)


def retrieve(
    q: dict, qi: int, chunks: list[dict], retriever: Retriever, args: argparse.Namespace, session: str, splitter: LLM | None = None
) -> tuple[list[dict], str, list[str], dict | None]:
    """检索并组装上下文。返回 (上下文块, 发给生成模型的用户消息, 子问题, 拆分调用的响应)。"""
    order, _ = retriever.rank(qi, tokenize(q["question"]))
    products = detect_products(q["question"])
    if products:
        order = [i for i in order if chunks[i]["doc_id"] in products]

    subs, split = [], None
    if splitter:
        split = splitter.chat(DECOMPOSE_PROMPTS[args.decompose], q["question"], session + "/split")
        subs = parse_subquestions(split["content"])
    if len(subs) >= SPLIT_MIN.get(args.decompose, 2):
        orders = [order]
        for s in subs:
            docs = detect_products(s) or products
            cos = retriever.chunk_vecs @ embed_query(s)
            orders.append([i for i in sorted(range(len(chunks)), key=lambda i: (-cos[i], i)) if not docs or chunks[i]["doc_id"] in docs])
        seeds = merge_seeds(orders, [args.k] + [args.sub_k] * len(subs))
        ctx = build_context(chunks, seeds, len(seeds), args.split_budget, args.expand, args.whole_clause)
        user = f"【条款片段】\n{render_context(chunks, ctx)}\n\n【问题】\n{q['question']}"
        if args.decompose == "d1":  # d1 的子问题是问句，交给生成模型逐一回答；d2 是检索词，不给
            listing = "\n".join(f"{n}. {s}" for n, s in enumerate(subs, 1))
            user += f"\n\n【问题拆分】（请逐一回答）\n{listing}"
    else:
        seeds = select_seeds(chunks, order, products, args.k, args.quota, args.neighbors)
        ctx = build_context(chunks, seeds, len(seeds), args.char_budget, args.expand, args.whole_clause)
        user = f"【条款片段】\n{render_context(chunks, ctx)}\n\n【问题】\n{q['question']}"
    return ctx, user, subs, split


def run_one(
    q: dict, qi: int, chunks: list[dict], retriever: Retriever, llm: LLM, args: argparse.Namespace, splitter: LLM | None = None
) -> dict:
    session = str(uuid.uuid5(uuid.NAMESPACE_URL, f"insurance-rag-eval/{args.run_name}/{q['id']}"))
    ctx, user, subs, split = retrieve(q, qi, chunks, retriever, args, session, splitter)
    resp = llm.chat(PROMPTS[args.prompt], user, session)
    usage, seconds, cached = dict(resp["usage"]), resp["seconds"], resp["cached"]
    if split:  # 拆分调用的 token 和耗时计入这道题
        usage = {k: usage[k] + split["usage"][k] for k in usage}
        seconds, cached = seconds + split["seconds"], cached and split["cached"]

    parsed = parse_json(resp["content"])
    answer = parsed["answer"] if parsed else resp["content"]
    refused = (parsed is not None and parsed.get("answerable") is False) or answer.strip().startswith(REFUSAL)

    cids = {item["cid"]: item for item in ctx}
    cited = list(dict.fromkeys(re.findall(r"\[(C\d+)\]", answer)))
    relevant_cids = [
        item["cid"]
        for item in ctx
        if any(is_relevant(chunks[j], s) for j in item["idxs"] for s in q["expected_sources"])
    ]
    return {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "answer": answer,
        "refused": refused,
        "parse_ok": parsed is not None,
        "finish_reason": resp.get("finish_reason"),
        "context": [{"cid": it["cid"], "chunk_ids": it["chunk_ids"]} for it in ctx],
        "context_chars": sum(chunks[j]["n_chars"] for it in ctx for j in it["idxs"]),
        "relevant_in_context": relevant_cids,
        "cited": cited,
        "invalid_citations": [c for c in cited if c not in cids],
        "cited_relevant": [c for c in cited if c in relevant_cids],
        "must_include": q["must_include"],
        "must_not_include_hits": [s for s in q.get("must_not_include", []) if s in answer],
        "subquestions": subs,
        "split_usage": split["usage"] if split else None,
        "usage": usage,
        "seconds": seconds,
        "cached": cached,
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
        "avg_context_chars": sum(r["context_chars"] for r in rows) / len(rows),
    }


def write_report(rows: list[dict], s: dict, args: argparse.Namespace, llm: LLM, n_chunks: int) -> Path:
    pct = lambda x: f"{x:.1%}"  # noqa: E731
    n_split = sum(len(r["subquestions"]) >= 2 for r in rows)
    split_desc = "关" if not args.decompose else (
        f"`{args.decompose}`（不带思考）；拆成 2 个及以上时，原问题前 {args.k} 块与每个子问题前 {args.sub_k} 块轮流合并，"
        f"字数预算 {args.split_budget}；拆开的题 {n_split} / {len(rows)}"
    )
    lines = [
        f"# 生成评测报告 · {args.run_name}",
        "",
        f"> 自动生成（{date.today()}）。评测集 `dataset/{args.golden}`。",
        "",
        "## 版本",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 模型 | `{llm.model}`（OpenCode Go，temperature=0，max_tokens={args.max_tokens or '模型上限'}） |",
        f"| 提示词 | `{args.prompt}`（`src/answer.py` 的 PROMPTS） |",
        f"| 检索 | 向量检索 bge-m3（`{args.emb}`）{'+ 重排 bge-reranker-v2-m3（前 ' + str(args.rerank_top) + ' 名）' if args.rerank else ''} + 产品过滤，文本块 {n_chunks} 个 |",
        f"| 种子 | 前 {args.k} 块；对比题按产品分配 {args.quota or '关'}；相邻条款 {args.neighbors or '关'} |",
        f"| 上下文 | 前 {args.k} 个种子块，{'同条款前后各扩展 ' + str(args.expand) + ' 块，' if args.expand else ''}去重，字数预算 {args.char_budget} |",
        f"| 整条条款 | {'条款不超过 ' + str(args.whole_clause) + ' 字就整条带上' if args.whole_clause else '关'} |",
        f"| 拆子问题 | {split_desc} |",
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
        f"| 平均上下文字数 | {s['avg_context_chars']:.0f} | |",
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
    ap.add_argument("--char-budget", type=int, default=8000)
    ap.add_argument("--expand", type=int, default=1, help="small-to-big：同条款前后各带几块，0 = gen_v1 的做法")
    ap.add_argument("--prompt", default="p2", choices=sorted(PROMPTS))
    ap.add_argument("--rerank", default=None, help="data/processed/rerank 下的分数目录；不设 = 纯向量检索")
    ap.add_argument("--rerank-top", type=int, default=20, help="向量检索前多少名进入重排")
    ap.add_argument("--quota", type=int, default=0, help="对比题每个产品各取几块，0 = 不分配")
    ap.add_argument("--neighbors", type=int, default=0, help="对前 N 个种子块带上相邻条款，0 = 不带")
    ap.add_argument("--decompose", default=None, choices=sorted(DECOMPOSE_PROMPTS), help="先拆子问题再多查询检索；不设 = 不拆")
    ap.add_argument("--sub-k", type=int, default=3, help="拆分后每个子问题取前几块")
    ap.add_argument("--split-budget", type=int, default=12000, help="拆成 2 个及以上子问题时的上下文字数预算")
    ap.add_argument("--whole-clause", type=int, default=0, help="条款不超过这么多字就整条带上；0 = 关（只带前后 expand 块）")
    ap.add_argument("--max-tokens", type=int, default=None, help="不设则用模型自己的最大输出上限")
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
    rerank_scores = None
    if args.rerank:
        rdir = ROOT / "data" / "processed" / "rerank" / args.rerank
        rmeta = json.loads((rdir / "meta.json").read_text(encoding="utf-8"))
        if rmeta["chunk_ids"] != meta["chunk_ids"] or rmeta["question_ids"] != meta["question_ids"]:
            raise SystemExit(f"{rdir} 与当前 chunks / golden 不一致，请重新运行 src/rerank.py")
        rerank_scores = np.load(rdir / "scores.npy")
    retriever = Retriever(
        "rerank" if args.rerank else "dense", index, np.load(emb / "chunks.npy"), np.load(emb / "questions.npy"),
        rerank_scores=rerank_scores, rerank_top=args.rerank_top,
    )
    llm = LLM(args.max_tokens)
    splitter = LLM(None, extra_body=NO_THINKING) if args.decompose else None

    todo = list(enumerate(golden))[: args.limit]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda p: run_one(p[1], p[0], chunks, retriever, llm, args, splitter), todo))

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
