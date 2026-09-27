"""路由：每道题决定用单次 RAG 还是 Agent。离线评估，不调用模型。

    uv run python src/route.py --name route_v1

两套系统在每个评测集上的答案和评委判定都已存在（reports/runs），
路由的端到端结果 = 按规则逐题选用其中一个答案；成本和延迟按实际会发生的调用累加。

路由规则只在开发集（golden_v2、golden_blind_v1）上设计，规则定下来后提交，再在验证集上跑。
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from statistics import median

from evaluate import REPORTS, ROOT, load_jsonl
from ops_report import cost, row_stats

PRODUCT_NAMES = {
    "cpic": r"太保|阿基米德",
    "taikang": r"泰康|惠嘉保",
    "chinalife": r"国寿|康宁尊享|中国人寿",
}
COMPARE = re.compile(r"哪个|哪款|区别|分别|比较|对比|谁的|谁更|相比|一样吗|差得多|三款|两款")
# 单次 RAG 自己承认材料不够的说法（在开发集的答案里看到的，定下后不再改）
INSUFFICIENT = re.compile(r"无法回答|无法判断|无法确定|片段(中)?未|片段(中)?没有|未提及|未涉及|未载明|未包含|未约定|没有提供|不足以")


def n_products(q: str) -> int:
    return sum(bool(re.search(p, q)) for p in PRODUCT_NAMES.values())


def n_subquestions(q: str) -> int:
    return len(re.findall(r"[？?]", q))


def shape_route(q: str) -> bool:
    """按问题形态：问到多款产品、带对比词，或一句话里有两个及以上问号 → Agent。"""
    return n_products(q) >= 2 or bool(COMPARE.search(q)) or n_subquestions(q) >= 2


def cascade_route(rag_row: dict) -> bool:
    """级联：单次 RAG 拒答，或答案里自己说材料不够 → 再交给 Agent。"""
    return rag_row["refused"] or bool(INSUFFICIENT.search(rag_row["answer"]))


POLICIES = ["always_rag", "always_agent", "shape", "cascade", "shape_or_cascade", "oracle"]
SCORE = {"correct": 2, "partial": 1, "incorrect": 0}

SETS = {
    "dev": [("golden_v2", "gen_v3_p2", "agent_v3_guard"), ("golden_blind_v1", "gen_v3_p2_blind", "agent_v3_guard_blind")],
    "v2.1+v3": [("golden_blind_v2.1", "gen_v3_p2_blind2_1", "agent_v3_guard_blind2_1"), ("golden_blind_v3", "gen_v3_p2_blind3", "agent_v3_guard_blind3")],
    # 独立检验集：路由 v1、v2 的规则都在它生成前提交（只加数据，不改规则）
    "v5": [("golden_blind_v5", "gen_v3_p2_blind5", "agent_v3_guard_blind5")],
}


def load_set(gold: str, rag: str, agent: str) -> list[dict]:
    golden = {q["id"]: q for q in load_jsonl(ROOT / "dataset" / f"{gold}.jsonl")}
    runs = {name: {r["id"]: r for r in load_jsonl(REPORTS / "runs" / f"{name}.jsonl")} for name in (rag, agent, f"judge_{rag}", f"judge_{agent}")}
    return [
        {"set": gold, "q": golden[i], "rag": runs[rag][i], "agent": runs[agent][i], "j_rag": runs[f"judge_{rag}"][i], "j_agent": runs[f"judge_{agent}"][i]}
        for i in golden
    ]


def decide(item: dict, policy: str) -> tuple[str, bool]:
    """返回 (采用哪个答案, 是否先跑了单次 RAG)。级联类策略总是先跑单次 RAG。"""
    q = item["q"]["question"]
    if policy == "always_rag":
        return "rag", True
    if policy == "always_agent":
        return "agent", False
    if policy == "shape":
        return ("agent", False) if shape_route(q) else ("rag", True)
    if policy == "cascade":
        return ("agent" if cascade_route(item["rag"]) else "rag"), True
    if policy == "shape_or_cascade":
        if shape_route(q):
            return "agent", False
        return ("agent" if cascade_route(item["rag"]) else "rag"), True
    if policy == "oracle":  # 事后按评委判定挑更好的一个；打平选便宜的单次 RAG。只作上限参考
        better = SCORE[item["j_agent"]["correctness"]] > SCORE[item["j_rag"]["correctness"]]
        return ("agent" if better else "rag"), not better
    raise ValueError(policy)


def evaluate(items: list[dict], policy: str, price: dict) -> dict:
    n_ans = sum(it["q"]["type"] != "refuse" for it in items)
    correct = partial = refuse_ok = faithful = to_agent = 0
    dollars, secs = 0.0, []
    for it in items:
        pick, ran_rag = decide(it, policy)
        j = it[f"j_{pick}"]
        if it["q"]["type"] == "refuse":
            refuse_ok += j["correctness"] == "correct"
        else:
            correct += j["correctness"] == "correct"
            partial += j["correctness"] == "partial"
        faithful += bool(j["faithful"])
        to_agent += pick == "agent"
        s = 0.0
        if ran_rag:
            dollars += cost(it["rag"]["usage"], price)
            s += row_stats(it["rag"], price)["seconds"]
        if pick == "agent":
            dollars += cost(it["agent"]["usage"], price)
            s += row_stats(it["agent"], price)["seconds"]
        secs.append(s)
    n = len(items)
    return {
        "correct": correct, "partial": partial, "n_ans": n_ans, "refuse_ok": refuse_ok, "n_ref": n - n_ans,
        "faithful": faithful, "n": n, "to_agent": to_agent, "usd_per_1k": dollars / n * 1000, "p50": median(secs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="route_v1")
    ap.add_argument("--sets", default="dev,v2.1+v3", help="逗号分隔，取自 SETS")
    args = ap.parse_args()
    price = json.loads((ROOT / "pricing.json").read_text(encoding="utf-8"))["models"]["deepseek-v4.1-flash"]

    lines = [
        f"# 路由评估 · {args.name}",
        "",
        f"> 自动生成（{date.today()}），`src/route.py`。离线评估：按规则逐题选用单次 RAG 或 Agent 已有的答案和评委判定，不调用模型。"
        "级联类策略的成本和延迟 = 单次 RAG + 被升级题的 Agent。oracle 是事后挑最优，只作上限参考。",
        "",
    ]
    for set_name in args.sets.split(","):
        items = [it for spec in SETS[set_name] for it in load_set(*spec)]
        lines += [
            f"## {set_name}（{len(items)} 题）",
            "",
            "| 策略 | 可回答题答对 | 部分正确 | 拒答题 | 忠实 | 走 Agent 的题 | 美元 / 千题 | 中位延迟（秒） |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for p in POLICIES:
            r = evaluate(items, p, price)
            lines.append(
                f"| {p} | {r['correct']}/{r['n_ans']} | {r['partial']} | {r['refuse_ok']}/{r['n_ref']} | {r['faithful']}/{r['n']} | "
                f"{r['to_agent']}/{r['n']} | {r['usd_per_1k']:.2f} | {r['p50']:.1f} |"
            )
        lines += ["", "逐题路由（shape / cascade 是否触发；两套系统的评委判定）：", "", "| 题 | 集合 | shape | cascade | 单次 RAG | Agent | 问题 |", "|---|---|---|---|---|---|---|"]
        for it in items:
            q = it["q"]
            lines.append(
                f"| {q['id']} | {it['set']} | {'✓' if shape_route(q['question']) else ''} | {'✓' if cascade_route(it['rag']) else ''} | "
                f"{it['j_rag']['correctness']} | {it['j_agent']['correctness']} | {q['question'][:40]} |"
            )
        lines.append("")
    out = REPORTS / f"{args.name}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(l for l in lines if l.startswith("## ") or l.startswith("| always") or l.startswith("| shape") or l.startswith("| cascade") or l.startswith("| oracle") or l.startswith("| 策略")))


if __name__ == "__main__":
    main()
