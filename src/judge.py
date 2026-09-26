"""LLM 评委：给生成答案打分，并用人工（或 AI）预先标注的小样本校准评委。

用法：
    uv run python src/judge.py --run gen_v3_p2 --golden golden_v2.jsonl
    uv run python src/judge.py --run gen_v3_p2_blind --golden golden_blind_v1.jsonl

两个维度分开判：
- 正确性（correct / partial / incorrect）：对照评测集的答案要点，不看上下文。
- 忠实度（faithful）：只对照生成时的上下文，不看标准答案。答案里每个事实都要有片段支持。
所以“检索失败后诚实拒答”是忠实但错误，“靠常识答对”是正确但不忠实。

评委与生成模型（DeepSeek）用不同家族（GLM），避免评委偏袒同家族模型的答案。
评委服务商由 .env 的 JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL 决定；没配就用 OpenCode Go 的 glm-5.3。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from answer import LLM, render_context
from evaluate import CHUNKS, GOLDEN, REPORTS, ROOT, load_jsonl

CALIB = ROOT / "dataset" / "judge_calib_v1.jsonl"
# 评委提示词只增不改，旧版本保留以便复现。
JUDGE_PROMPTS = {
    "j1": """你是保险条款问答系统的评测员。给你：问题、参考答案要点、系统看到的【条款片段】、系统的回答。请从两个维度独立打分。

一、正确性 correctness（只对照参考答案要点，不管片段）：
- 可回答题：
  - correct：覆盖了全部要点，且没有与要点矛盾的内容；
  - partial：覆盖了部分要点，或只回答了问题的一部分；
  - incorrect：没有覆盖任何要点、与要点矛盾，或拒答。
- 拒答题（题型为 refuse）：参考要点描述的是“资料里没有答案”时应有的表现。
  - correct：说明条款没有约定或无法回答，且没有编造具体答案。也包括“条款未约定X，产品只保障Y”这类有信息量的否定回答；
  - incorrect：给出了具体数额、天数、结论等参考要点认为资料里不存在的答案。
- 语义等价即算覆盖，不要求字面一致（如“十五日”与“15日”）。

二、忠实度 faithful（只对照条款片段，不管参考要点）：
- true：回答里每个事实陈述都能在片段中找到依据。拒答或“片段未提及”这类陈述，只要属实就算忠实；
- false：至少有一个事实陈述在片段中找不到依据，或与片段矛盾。把这些陈述列在 unsupported 里。

只输出一个 JSON 对象：
{"keypoints": [{"point": "要点", "covered": true或false}], "correctness": "correct|partial|incorrect", "faithful": true或false, "unsupported": ["无依据的陈述"], "reason": "一两句理由"}""",
    # j2：j1 的正确性规则有歧义（既覆盖部分要点又与另一要点矛盾时，评委会判 partial；
    # 拒答内容本身说中了“条款未规定”这类要点时，一律判 incorrect）。对抗测试和校准暴露后补两条优先规则。
    "j2": """你是保险条款问答系统的评测员。给你：问题、参考答案要点、系统看到的【条款片段】、系统的回答。请从两个维度独立打分。

一、正确性 correctness（只对照参考答案要点，不管片段）：
- 可回答题：
  - correct：覆盖了全部要点，且没有与要点矛盾的内容；
  - partial：覆盖了部分要点，或只回答了问题的一部分；
  - incorrect：没有覆盖任何要点、与要点矛盾，或拒答。
- 拒答题（题型为 refuse）：参考要点描述的是“资料里没有答案”时应有的表现。
  - correct：说明条款没有约定或无法回答，且没有编造具体答案。也包括“条款未约定X，产品只保障Y”这类有信息量的否定回答；
  - incorrect：给出了具体数额、天数、结论等参考要点认为资料里不存在的答案。
- 语义等价即算覆盖，不要求字面一致（如“十五日”与“15日”）。
- 优先规则一（矛盾优先）：回答中只要有一处与参考要点矛盾（数字、期限、比例写错，或结论相反），无论其他要点覆盖多少，一律判 incorrect。partial 只用于“有遗漏、但写出来的都没错”。
- 优先规则二（拒答看内容）：可回答题被拒答时，如果拒答里的说明本身覆盖了某个参考要点（例如要点就是“条款未规定X”，回答也指出片段未规定X），按覆盖情况判 partial 或 correct；否则判 incorrect。

二、忠实度 faithful（只对照条款片段，不管参考要点）：
- true：回答里每个事实陈述都能在片段中找到依据。拒答或“片段未提及”这类陈述，只要属实就算忠实；
- false：至少有一个事实陈述在片段中找不到依据，或与片段矛盾。把这些陈述列在 unsupported 里。

只输出一个 JSON 对象：
{"keypoints": [{"point": "要点", "covered": true或false}], "correctness": "correct|partial|incorrect", "faithful": true或false, "unsupported": ["无依据的陈述"], "reason": "一两句理由"}""",
}


def judge_one(
    row: dict, q: dict, chunks: list[dict], by_id: dict[str, int], llm: LLM, run: str, prompt: str = "j2"
) -> dict:
    ctx = [{"cid": c["cid"], "idxs": [by_id[x] for x in c["chunk_ids"]]} for c in row["context"]]
    user = (
        f"【题型】{q['type']}\n【问题】{q['question']}\n"
        f"【参考答案要点】\n" + "\n".join(f"- {p}" for p in q["must_include"]) + "\n\n"
        f"【条款片段】\n{render_context(chunks, ctx)}\n\n【系统回答】\n{row['answer']}"
    )
    session = str(uuid.uuid5(uuid.NAMESPACE_URL, f"insurance-rag-eval/judge/{run}/{q['id']}"))
    resp = llm.chat(JUDGE_PROMPTS[prompt], user, session)
    m = re.search(r"\{.*\}", resp["content"], re.S)
    try:
        v = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        v = None
    ok =isinstance(v, dict) and v.get("correctness") in ("correct", "partial", "incorrect")
    return {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "answer": row["answer"],
        "retrieval_ok": bool(row["relevant_in_context"]),
        "refused": row["refused"],
        "correctness": v["correctness"] if ok else None,
        "faithful": bool(v.get("faithful")) if ok else None,
        "unsupported": v.get("unsupported", []) if ok else [],
        "keypoints": v.get("keypoints", []) if ok else [],
        "reason": v.get("reason", "") if ok else resp["content"][:300],
        "parse_ok": ok,
        "usage": resp["usage"],
        "cached": resp["cached"],
    }


def attribute(r: dict) -> str:
    """答错时归因：证据没进上下文是检索问题，进了还答错是生成（或上下文不完整）问题。"""
    if r["correctness"] == "correct":
        return "正确"
    if r["type"] == "refuse":
        return "生成：该拒答未拒答"
    return "检索失败" if not r["retrieval_ok"] else "生成 / 上下文不完整"


def calibration(results: dict[str, dict], run: str) -> list[dict]:
    if not CALIB.exists():
        return []
    rows = []
    for lab in load_jsonl(CALIB):
        if lab["run"] == run and lab["id"] in results:
            j = results[lab["id"]]
            rows.append({**lab, "judge_correctness": j["correctness"], "judge_faithful": j["faithful"]})
    return rows


def write_report(res: list[dict], cal: list[dict], args: argparse.Namespace, llm: LLM) -> Path:
    n = len(res)
    cnt = Counter(r["correctness"] for r in res)
    ans = [r for r in res if r["type"] != "refuse"]
    ref = [r for r in res if r["type"] == "refuse"]
    pct = lambda a, b: f"{a / b:.1%}" if b else "-"  # noqa: E731
    lines = [
        f"# 评委报告 · {args.run}",
        "",
        f"> 自动生成（{date.today()}）。评委 `{llm.model}`（{llm.prefix}_* 配置，{llm.extra_body.get('reasoning_effort', '默认')} 思考强度），评委提示词 `{args.judge_prompt}`（`src/judge.py`）；被评的答案来自 `reports/runs/{args.run}.jsonl`。",
        "",
        "## 总体",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 正确 / 部分 / 错误 | {cnt['correct']} / {cnt['partial']} / {cnt['incorrect']}（共 {n}） |",
        f"| 可回答题正确率 | {pct(sum(r['correctness'] == 'correct' for r in ans), len(ans))}（部分正确 {sum(r['correctness'] == 'partial' for r in ans)} 题） |",
        f"| 拒答题正确率 | {pct(sum(r['correctness'] == 'correct' for r in ref), len(ref))} |",
        f"| 忠实度 | {pct(sum(bool(r['faithful']) for r in res), n)} |",
        f"| 评委输出解析失败 | {sum(not r['parse_ok'] for r in res)} |",
        "",
        "## 失败归因",
        "",
        "| 归因 | 题数 | 题目 |",
        "|---|---|---|",
    ]
    groups: dict[str, list[str]] = {}
    for r in res:
        groups.setdefault(attribute(r), []).append(r["id"] + ("（部分）" if r["correctness"] == "partial" else ""))
    for k, ids in groups.items():
        if k != "正确":
            lines.append(f"| {k} | {len(ids)} | {'、'.join(ids)} |")
    if cal:
        agree_c = sum(c["correctness"] == c["judge_correctness"] for c in cal)
        agree_f = sum(c["faithful"] == c["judge_faithful"] for c in cal)
        lines += [
            "",
            "## 校准（对照事先锁定的标注 `dataset/judge_calib_v1.jsonl`）",
            "",
            f"- 正确性一致：{agree_c}/{len(cal)}",
            f"- 忠实度一致：{agree_f}/{len(cal)}",
            "",
            "| 题 | 标注 | 评委 | 忠实度 标注/评委 | 评委理由 |",
            "|---|---|---|---|---|",
        ]
        for c in cal:
            mark = "" if c["correctness"] == c["judge_correctness"] else " ⚠"
            reason = results_reason(res, c["id"]).replace("|", "／").replace("\n", " ")
            lines.append(
                f"| {c['id']}{mark} | {c['correctness']} | {c['judge_correctness']} | {c['faithful']} / {c['judge_faithful']} | {reason} |"
            )
    lines += ["", "## 逐题", "", "| 题 | 类型 | 正确性 | 忠实 | 无依据陈述 | 理由 |", "|---|---|---|---|---|---|"]
    for r in res:
        uns = "；".join(r["unsupported"]).replace("|", "／")
        lines.append(
            f"| {r['id']} | {r['type']} | {r['correctness']} | {r['faithful']} | {uns} | {r['reason'].replace('|', '／').replace(chr(10), ' ')} |"
        )
    path = REPORTS / f"judge_{args.run}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def results_reason(res: list[dict], qid: str) -> str:
    return next(r["reason"] for r in res if r["id"] == qid)


def judge_llm(max_tokens: int | None, model: str | None = None) -> LLM:
    """.env 里配了 JUDGE_BASE_URL 就用评委专用的服务商，否则沿用生成的 OpenCode Go。"""
    load_dotenv(ROOT / ".env")
    if os.environ.get("JUDGE_BASE_URL"):
        # GLM-5.3 思考强度 low / high / max；评委要判得严，显式设为最高档（官方默认也是 max，写明便于复现）
        effort = os.environ.get("JUDGE_REASONING_EFFORT", "max")
        extra = {"thinking": {"type": "enabled"}, "reasoning_effort": effort}
        return LLM(max_tokens, model=model, prefix="JUDGE", extra_body=extra)
    return LLM(max_tokens, model=model or "glm-5.3")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="reports/runs 下的生成结果名")
    ap.add_argument("--golden", default=GOLDEN.name)
    ap.add_argument("--judge-model", default=None, help="默认读 JUDGE_MODEL；没配 JUDGE_* 时用 OpenCode Go 的 glm-5.3")
    ap.add_argument("--max-tokens", type=int, default=None, help="不设则用模型自己的最大输出上限")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--judge-prompt", default="j2", choices=sorted(JUDGE_PROMPTS))
    args = ap.parse_args()

    chunks = load_jsonl(CHUNKS)
    by_id = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    golden = {q["id"]: q for q in load_jsonl(GOLDEN.parent / args.golden)}
    rows = load_jsonl(REPORTS / "runs" / f"{args.run}.jsonl")
    llm = judge_llm(args.max_tokens, args.judge_model)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        res = list(pool.map(lambda r: judge_one(r, golden[r["id"]], chunks, by_id, llm, args.run, args.judge_prompt), rows))

    with (REPORTS / "runs" / f"judge_{args.run}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cal = calibration({r["id"]: r for r in res}, args.run)
    report = write_report(res, cal, args, llm)
    cnt = Counter(r["correctness"] for r in res)
    print(
        f"{args.run}: {dict(cnt)} faithful={sum(bool(r['faithful']) for r in res)}/{len(res)} "
        f"parse_fail={sum(not r['parse_ok'] for r in res)} calib={sum(c['correctness'] == c['judge_correctness'] for c in cal)}/{len(cal)} "
        f"api_calls={sum(not r['cached'] for r in res)}"
    )
    print(f"report -> {report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
