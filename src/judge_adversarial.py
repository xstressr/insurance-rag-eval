"""评委对抗测试：把正确答案人为改坏，看评委能不能识别。

校准集里的答案全是忠实的，只能证明评委不会冤枉好答案，证明不了它能抓到编造。
这里构造两类坏答案：
- tamper：改错关键数字或结论 → 应判 incorrect 且 faithful=false；
- inject：保留原答案，追加一句片段里没有的内容 → 应判 faithful=false。

用法：uv run python src/judge_adversarial.py
"""

from __future__ import annotations

import json

from evaluate import CHUNKS, GOLDEN, REPORTS, load_jsonl
from judge import judge_llm, judge_one

# (题号, 类型, 原文片段, 替换为)；inject 的原文片段为空表示追加到末尾
CASES = [
    ("q001", "tamper", "15日", "30日"),
    ("q012", "tamper", "125种", "150种"),
    ("q020", "tamper", "不赔。", "赔。遗传性疾病导致的重疾在保障范围内，"),
    ("q003", "inject", "", "等待期内确诊重疾的，也可以申请延期承保，待等待期满后再给付重大疾病保险金 [C3]。"),
    ("q030", "inject", "", "在该期间内退保，保险公司还会按同期银行存款利率额外支付利息 [C5]。"),
    ("q007", "inject", "", "此外，轻症疾病保险金给付后，后续保险费可以减半交纳 [C1]。"),
]
# 留出集：j2 是看过上面 6 个案例后才写的，这 5 个在 j2 定稿后才构造，用来检验 j2 是否通用。
# omit：只保留答案开头到指定片段为止，制造“有遗漏但没写错”，期望 partial 且忠实（检验 j2 不会矫枉过正）。
HOLDOUT = [
    ("q004", "tamper", "80%", "90%"),
    ("q006", "tamper", "三次", "五次"),
    ("q016", "tamper", "二级合格或者二级合格以上", "三级合格或者三级合格以上"),
    ("q009", "inject", "", "超过55周岁的，可以加费承保至60周岁 [C1]。"),
    ("q002", "omit", "有90 日的等待期 [C1]。", ""),
]


def main() -> None:
    chunks = load_jsonl(CHUNKS)
    by_id = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    golden = {q["id"]: q for q in load_jsonl(GOLDEN.parent / "golden_v2.jsonl")}
    rows = {r["id"]: r for r in load_jsonl(REPORTS / "runs" / "gen_v3_p2.jsonl")}
    llm = judge_llm(None)

    lines = [
        "# 评委对抗测试 · judge_adversarial_v1",
        "",
        f"> 把 gen_v3_p2 中的正确答案人为改坏，检验评委 `{llm.model}`（{llm.prefix}_* 配置，提示词 j2）能否识别。",
        "> tamper：改错关键数字或结论，期望 incorrect + 不忠实；inject：追加片段里没有的内容，期望不忠实；omit：截掉部分要点，期望 partial + 忠实。",
        "",
        "| 题 | 类型 | 改动 | 评委正确性 | 评委忠实 | 识别成功 | 评委指出的无依据陈述 |",
        "|---|---|---|---|---|---|---|",
    ]
    ok_n = 0
    cases = [(c, "已见") for c in CASES] + [(c, "留出") for c in HOLDOUT]
    lines[-2] = "| 集合 | 题 | 类型 | 改动 | 评委正确性 | 评委忠实 | 识别成功 | 评委指出的无依据陈述 |"
    lines[-1] = "|---|---|---|---|---|---|---|---|"
    hold_ok = 0
    for (qid, kind, old, new), split in cases:
        row = dict(rows[qid])
        if kind == "tamper":
            assert old in row["answer"], (qid, old)
            row["answer"] = row["answer"].replace(old, new, 1)
        elif kind == "omit":
            assert old in row["answer"], (qid, old)
            row["answer"] = row["answer"][: row["answer"].index(old) + len(old)]
        else:
            row["answer"] = row["answer"].rstrip() + new
        r = judge_one(row, golden[qid], chunks, by_id, llm, "adversarial")
        if kind == "omit":
            ok = r["faithful"] is True and r["correctness"] == "partial"
        else:
            ok = r["faithful"] is False and (kind == "inject" or r["correctness"] == "incorrect")
        ok_n += ok
        hold_ok += ok and split == "留出"
        change = {"tamper": f"{old} → {new}", "inject": f"追加：{new}", "omit": f"截断到：{old}"}[kind]
        uns = "；".join(r["unsupported"]).replace("|", "／")
        lines.append(f"| {split} | {qid} | {kind} | {change} | {r['correctness']} | {r['faithful']} | {'是' if ok else '**否**'} | {uns} |")
        print(qid, kind, r["correctness"], r["faithful"], "OK" if ok else "MISS")
    lines += ["", f"识别成功 {ok_n}/{len(cases)}，其中留出集 {hold_ok}/{len(HOLDOUT)}。"]
    (REPORTS / f"judge_adversarial_v1_{llm.model}_j2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{ok_n}/{len(cases)} holdout {hold_ok}/{len(HOLDOUT)}")


if __name__ == "__main__":
    main()
