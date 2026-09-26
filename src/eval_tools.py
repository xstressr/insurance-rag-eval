"""工具任务评测：Agent 能否正确组合 policy_lookup / benefit_calc / date_calc 与条款检索，给出对的金额或日期。

    uv run python src/eval_tools.py --run-name tools_v1

标准答案在 dataset/tool_tasks_v1.jsonl，由人工推算（不是用计算工具生成的）。评分全部由程序完成：
- 结果正确：应赔金额 / 日期出现在答案里；应赔 0 的题，答案要明确说不赔并说出原因（未投保 / 已达上限）；
- 工具使用：required_tools 里的工具都被成功调用过（没有被校验拦截、没有返回错误）；
- 另外统计工具调用成功率、每题调用次数、护栏退回次数。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from agent import Corpus
from agent_graph import run_episode
from answer import LLM
from evaluate import REPORTS, ROOT, load_jsonl

TASKS = ROOT / "dataset" / "tool_tasks_v1.jsonl"
NO_PAY = ["不赔", "不能赔", "不予", "无法获得", "不承担", "不属于", "不会赔", "不能获得", "0元", "0 元", "不再给付", "不能再"]
REASON = {"未投保": ["未投保", "没有投保", "没投保", "未购买", "未选择"], "上限": ["上限", "四次", "4次", "4 次", "已达"]}


def amounts(text: str) -> set[int]:
    """抽取金额：150000、150,000、15万、15.0万元。"""
    out = set()
    for num, wan in re.findall(r"(\d[\d,]*(?:\.\d+)?)\s*(万)?", text):
        try:
            v = float(num.replace(",", ""))
        except ValueError:
            continue
        out.add(round(v * 10000) if wan else round(v))
    return out


def dates(text: str) -> set[str]:
    out = set(re.findall(r"\d{4}-\d{2}-\d{2}", text))
    for y, m, d in re.findall(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text):
        out.add(f"{y}-{int(m):02d}-{int(d):02d}")
    return out


def score(task: dict, row: dict) -> dict:
    ans = row["answer"]
    exp = task["expected"]
    if isinstance(exp, str):
        ok = exp in dates(ans)
    elif exp == 0:
        ok = any(p in ans for p in NO_PAY) and any(k in ans for k in REASON.get(task["expected_keyword"], [""]))
    else:
        ok = exp in amounts(ans)
    calls = [t for t in row["trajectory"] if t["tool"]]
    good = {t["tool"] for t in calls if not t.get("error") and not str(t.get("result_head", "")).startswith("错误")}
    return {
        "correct": ok,
        "tools_ok": all(t in good for t in task["required_tools"]),
        "missing_tools": [t for t in task["required_tools"] if t not in good],
        "n_calls": len(calls),
        "n_failed_calls": sum(bool(t.get("error")) or str(t.get("result_head", "")).startswith("错误") for t in calls),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--agent-prompt", default="a3")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--guard", type=int, default=1)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    args.toolset, args.engine = "full", "LangGraph"

    tasks = load_jsonl(TASKS)
    qs = [{**t, "type": "tool", "expected_sources": [], "must_include": [t["derivation"]]} for t in tasks]
    corpus, llm = Corpus(), LLM(None)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda q: run_episode(q, corpus, llm, args), qs))
    results = [{**r, **score(t, r)} for t, r in zip(tasks, rows)]

    with (REPORTS / "runs" / f"{args.run_name}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(results)
    calls = sum(r["n_calls"] for r in results)
    failed = sum(r["n_failed_calls"] for r in results)
    lines = [
        f"# 工具任务报告 · {args.run_name}",
        "",
        f"> 自动生成（{date.today()}）。任务 `dataset/tool_tasks_v1.jsonl`（{n} 题，标准答案人工推算）。LangGraph Agent，提示词 `{args.agent_prompt}`，全部工具，最多 {args.max_steps} 轮，护栏退回上限 {args.guard}。",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 结果正确 | {sum(r['correct'] for r in results)}/{n} |",
        f"| 必要工具都调用成功 | {sum(r['tools_ok'] for r in results)}/{n} |",
        f"| 工具调用成功率 | {(calls - failed) / calls:.1%}（{calls - failed}/{calls}） |",
        f"| 平均工具调用 | {calls / n:.1f} |",
        f"| 结束方式 | {dict(Counter(r['stop'] for r in results))} |",
        f"| 护栏退回 | {sum(r['guard_rejections'] for r in results)} |",
        f"| 输入 / 输出 token | {sum(r['usage']['prompt'] for r in results)} / {sum(r['usage']['completion'] for r in results)} |",
        "",
        "| 题 | 标准答案 | 正确 | 工具 | 调用序列 | 答案 |",
        "|---|---|---|---|---|---|",
    ]
    for t, r in zip(tasks, results):
        seq = " → ".join(
            f"{x['tool']}{'✗' if x.get('error') or str(x.get('result_head', '')).startswith('错误') else ''}" for x in r["trajectory"] if x["tool"]
        )
        ans = r["answer"].replace("\n", " ").replace("|", "／")
        lines.append(
            f"| {t['id']} | {t['expected']} | {'✓' if r['correct'] else '✗'} | {'✓' if r['tools_ok'] else '缺 ' + '、'.join(r['missing_tools'])} | {seq} | {ans} |"
        )
    (REPORTS / f"{args.run_name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{args.run_name}: correct {sum(r['correct'] for r in results)}/{n}, tools_ok {sum(r['tools_ok'] for r in results)}/{n}, call success {(calls - failed)}/{calls}")


if __name__ == "__main__":
    main()
