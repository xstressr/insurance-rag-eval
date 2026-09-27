"""路由 v2：用便宜的模型做判断。route.py（v1）已锁定，这里单独实现。

    uv run python src/route_llm.py --name route_v2_design --sets dev,v2.1+v3

两种判断，都用 deepseek-v4.1-flash，结果缓存：
- pre（先分流）：只看问题，判断是否需要多步查找 → 直接走 Agent，否则走单次 RAG；
- post（后检查）：先跑单次 RAG，再判断答案是否完整回答了每个子问题 → 不完整就升级给 Agent。
判断调用的 token 和耗时计入成本与延迟。其余与 route.py 相同：逐题选用已有答案和评委判定。
"""

from __future__ import annotations

import argparse
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from statistics import median

import route
from answer import LLM
from evaluate import REPORTS, ROOT
from ops_report import cost, row_stats

PRE_PROMPT = """你是保险条款问答系统的分流器。系统有两条路径：
- rag：检索一次，适合只问一件事、一条条款就能回答的问题；
- agent：可以多次检索、按条款号读原文，适合需要多步查找的问题。

满足下列任一条件，选 agent：
1. 问题里有两个或以上需要分别回答的子问题（哪怕只用一个问号，例如“可以吗，要怎么办”）；
2. 需要对比两款或以上产品；
3. 需要把两条以上条款结合起来推理（例如先查某个定义，再判断能不能赔）。
否则选 rag。

只输出 JSON：{"route": "rag" 或 "agent", "reason": "一句话"}"""

POST_PROMPT = """你检查保险条款问答系统的答案是否完整。答案只依据检索到的条款片段，你看不到片段。

判为不完整（complete=false）的情况：
1. 问题里有子问题，答案完全没有回应；
2. 答案对某个子问题说“片段未提及 / 无法回答 / 未约定”，而这类内容是保险条款通常会写的（如等待期、宽限期、犹豫期、退保与现金价值、受益人、保险事故通知、理赔材料、保障范围、责任免除、疾病定义、投保年龄、保费豁免、合同效力），说明很可能是检索没找到。

判为完整（complete=true）的情况：
1. 每个子问题都有依据地回答了；
2. 答案说“条款未写”的部分，本来就是条款通常不写的内容（如具体费率和保费、核保标准、理赔率、公司经营、线上或柜台等办理渠道、税务）。

只输出 JSON：{"complete": true 或 false, "reason": "一句话"}"""


def judge_call(llm: LLM, system: str, user: str, key: str) -> dict:
    session = str(uuid.uuid5(uuid.NAMESPACE_URL, f"insurance-rag-eval/route_v2/{key}"))
    resp = llm.chat(system, user, session)
    try:
        obj = json.loads(resp["content"][resp["content"].index("{") : resp["content"].rindex("}") + 1])
    except ValueError:
        obj = {}
    return {"obj": obj, "usage": resp["usage"], "seconds": resp["seconds"], "cached": resp["cached"]}


def annotate(items: list[dict], llm: LLM, workers: int = 4) -> None:
    """给每题补上 pre 和 post 两个判断（结果缓存，重跑不再调用）。"""

    def one(it: dict) -> None:
        q = it["q"]["question"]
        it["pre"] = judge_call(llm, PRE_PROMPT, f"问题：{q}", f"pre/{it['set']}/{it['q']['id']}")
        it["post"] = judge_call(llm, POST_PROMPT, f"问题：{q}\n\n答案：{it['rag']['answer']}", f"post/{it['set']}/{it['q']['id']}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, items))


def pre_agent(it: dict) -> bool:
    return it["pre"]["obj"].get("route") == "agent"


def post_escalate(it: dict) -> bool:
    return it["post"]["obj"].get("complete") is False


def decide(it: dict, policy: str) -> tuple[str, bool, list[str]]:
    """返回 (采用哪个答案, 是否跑了单次 RAG, 用到的判断调用)。"""
    if policy in ("always_rag", "always_agent", "oracle", "cascade_v1"):
        pick, ran = route.decide(it, "cascade" if policy == "cascade_v1" else policy)
        return pick, ran, []
    if policy == "llm_pre":
        return ("agent", False, ["pre"]) if pre_agent(it) else ("rag", True, ["pre"])
    if policy == "llm_post":
        return ("agent" if post_escalate(it) else "rag"), True, ["post"]
    if policy == "llm_pre_then_post":  # 先分流；分到 rag 的再做答案检查
        if pre_agent(it):
            return "agent", False, ["pre"]
        return ("agent" if post_escalate(it) else "rag"), True, ["pre", "post"]
    raise ValueError(policy)


POLICIES = ["always_rag", "always_agent", "cascade_v1", "llm_pre", "llm_post", "llm_pre_then_post", "oracle"]


def evaluate(items: list[dict], policy: str, price: dict) -> dict:
    n_ans = sum(it["q"]["type"] != "refuse" for it in items)
    correct = partial = refuse_ok = to_agent = 0
    dollars, secs = 0.0, []
    for it in items:
        pick, ran_rag, calls = decide(it, policy)
        j = it[f"j_{pick}"]
        if it["q"]["type"] == "refuse":
            refuse_ok += j["correctness"] == "correct"
        else:
            correct += j["correctness"] == "correct"
            partial += j["correctness"] == "partial"
        to_agent += pick == "agent"
        s = 0.0
        for c in calls:
            dollars += cost(it[c]["usage"], price)
            s += it[c]["seconds"]
        if ran_rag:
            dollars += cost(it["rag"]["usage"], price)
            s += row_stats(it["rag"], price)["seconds"]
        if pick == "agent":
            dollars += cost(it["agent"]["usage"], price)
            s += row_stats(it["agent"], price)["seconds"]
        secs.append(s)
    n = len(items)
    return {"correct": correct, "partial": partial, "n_ans": n_ans, "refuse_ok": refuse_ok, "n_ref": n - n_ans,
            "n": n, "to_agent": to_agent, "usd_per_1k": dollars / n * 1000, "p50": median(secs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="route_v2")
    ap.add_argument("--sets", default="dev,v2.1+v3")
    args = ap.parse_args()
    price = json.loads((ROOT / "pricing.json").read_text(encoding="utf-8"))["models"]["deepseek-v4.1-flash"]
    llm = LLM(None)
    lines = [
        f"# 路由评估 · {args.name}",
        "",
        f"> 自动生成（{date.today()}），`src/route_llm.py`。判断模型 `{llm.model}`。"
        "离线评估：按规则逐题选用已有的答案和评委判定；判断调用、单次 RAG、Agent 的成本和延迟按实际会发生的调用累加。oracle 只作上限参考。",
        "",
    ]
    for set_name in args.sets.split(","):
        items = [it for spec in route.SETS[set_name] for it in route.load_set(*spec)]
        annotate(items, llm)
        lines += [f"## {set_name}（{len(items)} 题）", "", "| 策略 | 可回答题答对 | 部分正确 | 拒答题 | 走 Agent 的题 | 美元 / 千题 | 中位延迟（秒） |", "|---|---|---|---|---|---|---|"]
        for p in POLICIES:
            r = evaluate(items, p, price)
            lines.append(f"| {p} | {r['correct']}/{r['n_ans']} | {r['partial']} | {r['refuse_ok']}/{r['n_ref']} | {r['to_agent']}/{r['n']} | {r['usd_per_1k']:.2f} | {r['p50']:.1f} |")
        lines += ["", "| 题 | 集合 | pre | post | 单次 RAG | Agent | pre 理由 | post 理由 |", "|---|---|---|---|---|---|---|---|"]
        for it in items:
            lines.append(
                f"| {it['q']['id']} | {it['set']} | {it['pre']['obj'].get('route', '?')} | {'不完整' if post_escalate(it) else '完整'} | "
                f"{it['j_rag']['correctness']} | {it['j_agent']['correctness']} | {str(it['pre']['obj'].get('reason', ''))[:40]} | {str(it['post']['obj'].get('reason', ''))[:50]} |"
            )
        lines.append("")
    (REPORTS / f"{args.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(l for l in lines if l.startswith("## ") or (l.startswith("| ") and l.split("|")[1].strip() in POLICIES + ["策略"])))


if __name__ == "__main__":
    main()
