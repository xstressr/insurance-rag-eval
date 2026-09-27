"""拆子问题（gen_v4）与单次 RAG（gen_v3_p2）、Agent（agent_v3_guard）的端到端对比。

    uv run python src/decompose_report.py --name decompose_dev --variant d2 --sets dev

打分、成本、延迟的算法与 route.py 相同（always_rag / always_agent 两个策略），只是把单次 RAG 换成拆分后的版本。
"""

from __future__ import annotations

import argparse
import json
from datetime import date

import route
from evaluate import REPORTS, ROOT

# (集合, 标注文件, 后缀)；run 名 = gen_v3_p2{后缀} / gen_v4_{variant}{后缀} / agent_v3_guard{后缀}
SETS = {
    "dev": [
        ("golden_v2", "golden_v2", ""),
        ("blind_v1", "golden_blind_v1", "_blind"),
        ("blind_v2.1", "golden_blind_v2.1", "_blind2_1"),
        ("blind_v3", "golden_blind_v3", "_blind3"),
        ("blind_v5", "golden_blind_v5", "_blind5"),
    ],
    "v6": [("blind_v6", "golden_blind_v6", "_blind6")],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="decompose_dev")
    ap.add_argument("--variant", default="d2")
    ap.add_argument("--sets", default="dev")
    args = ap.parse_args()
    price = json.loads((ROOT / "pricing.json").read_text(encoding="utf-8"))["models"]["deepseek-v4.1-flash"]

    lines = [
        f"# 拆子问题 · {args.name}",
        "",
        f"> 自动生成（{date.today()}），`src/decompose_report.py`。评委 j2；成本按列表价，含拆分调用；延迟是每题各次调用耗时之和。",
        "",
        "| 集合 | 系统 | 可回答题答对 | 部分正确 | 拒答题 | 忠实 | 美元 / 千题 | 中位延迟（秒） |",
        "|---|---|---|---|---|---|---|---|",
    ]
    detail = ["", "## 逐题（评委判定）", "", "| 集合 | 题 | 单次 RAG | 拆分 | Agent | 拆分的检索词 |", "|---|---|---|---|---|---|"]
    for label, gold, suffix in SETS[args.sets]:
        systems = {
            "单次 RAG": route.load_set(gold, f"gen_v3_p2{suffix}", f"agent_v3_guard{suffix}"),
            f"拆分 {args.variant}": route.load_set(gold, f"gen_v4_{args.variant}{suffix}", f"agent_v3_guard{suffix}"),
        }
        rows = [(name, route.evaluate(items, "always_rag", price)) for name, items in systems.items()]
        rows.append(("Agent", route.evaluate(systems["单次 RAG"], "always_agent", price)))
        for name, r in rows:
            lines.append(
                f"| {label} | {name} | {r['correct']}/{r['n_ans']} | {r['partial']} | {r['refuse_ok']}/{r['n_ref']} | "
                f"{r['faithful']}/{r['n']} | {r['usd_per_1k']:.2f} | {r['p50']:.1f} |"
            )
        for base, new in zip(*systems.values()):
            subs = "；".join(new["rag"].get("subquestions") or []) or "—"
            detail.append(
                f"| {label} | {base['q']['id']} | {base['j_rag']['correctness']} | {new['j_rag']['correctness']} | "
                f"{base['j_agent']['correctness']} | {subs} |"
            )
    out = REPORTS / f"{args.name}.md"
    out.write_text("\n".join(lines + detail) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines[4:]))


if __name__ == "__main__":
    main()
