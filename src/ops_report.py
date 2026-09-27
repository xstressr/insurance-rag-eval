"""运行指标汇总：延迟、token、成本、错误率。只读 reports/runs 下已有的结果，不调用模型。

    uv run python src/ops_report.py --name ops_v1

延迟：每次模型调用的耗时存在缓存里，回放时读出的是当初真实调用的耗时，所以缓存回放也能统计延迟。
Agent 的端到端延迟 = 各次模型调用耗时之和 + 工具耗时；单次 RAG 只有一次模型调用（查询向量是离线算好的，不计入）。
成本：token × pricing.json 里的参考单价。本项目实际走订阅套餐，这是“按量调用官方 API”的等价成本。
"""

from __future__ import annotations

import argparse
import json
from datetime import date

from evaluate import REPORTS, ROOT, load_jsonl

DEFAULT_RUNS = [
    ("单次 RAG · golden", "gen_v3_p2"),
    ("单次 RAG · blind", "gen_v3_p2_blind"),
    ("Agent a2 + 护栏 · golden", "agent_v3_guard"),
    ("Agent a2 + 护栏 · blind", "agent_v3_guard_blind"),
    ("单次 RAG · blind_v2", "gen_v3_p2_blind2"),
    ("Agent a2 + 护栏 · blind_v2", "agent_v3_guard_blind2"),
    ("单次 RAG · blind_v3", "gen_v3_p2_blind3"),
    ("Agent a2 + 护栏 · blind_v3", "agent_v3_guard_blind3"),
    ("Agent a3 + 计算工具 · 工具任务", "tools_v2"),
    ("Agent a3 + 权限 · 安全用例", "security_v2"),
]


def pct(xs: list[float], p: float) -> float:
    """最近秩法分位数：样本少时不插值，报告的都是真实出现过的值。"""
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, round(p * len(xs) + 0.5) - 1))]


def cost(usage: dict, price: dict) -> float:
    """美元。思考 token 已包含在 completion 里，不重复计。"""
    return (usage["prompt"] * price["input"] + usage["completion"] * price["output"]) / 1e6


def row_stats(r: dict, price: dict) -> dict:
    trace = r.get("trace")
    if trace is not None:  # Agent
        llm = [s for s in trace if s["kind"] == "llm"]
        tools = [s for s in trace if s["kind"] == "tool"]
        llm_s = sum(s["seconds"] for s in llm)
        tool_s = sum(s["ms"] for s in tools) / 1000
        truncated = any(s["finish_reason"] == "length" for s in llm)
        abnormal = r["stop"] != "final_answer"
    else:  # 单次 RAG
        llm, tools, llm_s, tool_s = [None], [], r["seconds"], 0.0
        truncated = r.get("finish_reason") == "length"
        abnormal = not r["parse_ok"]
    return {
        "seconds": llm_s + tool_s,
        "llm_seconds": llm_s,
        "llm_calls": len(llm),
        "tool_calls": len(tools),
        "tool_invalid": r.get("n_tool_errors", 0),  # 参数校验拦截
        "tool_denied": len(r.get("denials", [])),  # 权限拒绝（安全用例里是预期行为）
        "prompt": r["usage"]["prompt"],
        "completion": r["usage"]["completion"],
        "cost": cost(r["usage"], price),
        "empty": not r["answer"].strip(),
        "truncated": truncated,
        "abnormal": abnormal,
        "guard": r.get("guard_rejections", 0),
        "residual": bool(r.get("residual_severe")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ops_v1")
    args = ap.parse_args()
    prices = json.loads((ROOT / "pricing.json").read_text(encoding="utf-8"))["models"]
    gen_price, judge_price = prices["deepseek-v4.1-flash"], prices["glm-5.3"]

    table, err_table, judge_table, notes = [], [], [], []
    for label, run in DEFAULT_RUNS:
        rows = load_jsonl(REPORTS / "runs" / f"{run}.jsonl")
        if "trace" not in rows[0] and "steps" in rows[0]:
            notes.append(f"`{run}` 没有追踪字段，需要用当前代码回放一次。")
            continue
        st = [row_stats(r, gen_price) for r in rows]
        n = len(st)
        secs = [s["seconds"] for s in st]
        calls = sum(s["tool_calls"] for s in st)
        tool_share = 1 - sum(s["llm_seconds"] for s in st) / max(sum(secs), 1e-9)
        table.append(
            f"| {label} | {n} | {pct(secs, 0.5):.1f} | {pct(secs, 0.95):.1f} | {max(secs):.1f} | {tool_share:.1%} | "
            f"{sum(s['llm_calls'] for s in st) / n:.1f} | {sum(s['prompt'] for s in st) / n:,.0f} / {sum(s['completion'] for s in st) / n:,.0f} | "
            f"{sum(s['cost'] for s in st) / n * 1000:.2f} |"
        )
        err_table.append(
            f"| {label} | {sum(s['empty'] for s in st)} | {sum(s['truncated'] for s in st)} | {sum(s['abnormal'] for s in st)} | "
            f"{sum(s['tool_invalid'] for s in st)} / {calls} | {sum(s['tool_denied'] for s in st)} | "
            f"{sum(s['guard'] for s in st)} | {sum(s['residual'] for s in st)} |"
        )
        jpath = REPORTS / "runs" / f"judge_{run}.jsonl"
        if jpath.exists():
            js = load_jsonl(jpath)
            jc = sum(cost(j["usage"], judge_price) for j in js)
            jsecs = [j["seconds"] for j in js if j.get("seconds") is not None]
            reasoning = sum(j["usage"]["reasoning"] for j in js) / max(sum(j["usage"]["completion"] for j in js), 1)
            judge_table.append(
                f"| {label} | {len(js)} | {sum(not j['parse_ok'] for j in js)} | {sum(j['usage']['prompt'] for j in js) / len(js):,.0f} / "
                f"{sum(j['usage']['completion'] for j in js) / len(js):,.0f}（思考占 {reasoning:.0%}） | "
                f"{(f'{pct(jsecs, 0.5):.0f}' if jsecs else '未记录')} | {jc / len(js) * 1000:.2f} |"
            )

    lines = [
        f"# 运行指标 · {args.name}",
        "",
        f"> 自动生成（{date.today()}），`src/ops_report.py`。只读已有运行结果，不调用模型。单价见 `pricing.json`：生成 DeepSeek V4.1 Flash "
        f"${gen_price['input']} / ${gen_price['output']}，评委 GLM-5.3 ${judge_price['input']} / ${judge_price['output']}（美元 / 百万 token，输入 / 输出）。",
        "",
        "## 延迟与成本（每题）",
        "",
        "延迟是模型调用的原始耗时（缓存里记录的）加工具耗时，单位秒。成本是按量调用官方 API 的等价成本，单位美元 / 千题。",
        "",
        "| 配置 | 题数 | P50 | P95 | 最慢 | 工具耗时占比 | 模型调用次数 | 输入 / 输出 token | 美元 / 千题 |",
        "|---|---|---|---|---|---|---|---|---|",
        *table,
        "",
        "## 错误与异常",
        "",
        "- 空答案：最终没有答案文字。截断：有模型调用因输出上限被截断。",
        "- 未正常结束：Agent 不是通过 final_answer 提交（强制作答、直接输出文字、步数耗尽）；单次 RAG 是 JSON 解析失败。",
        "- 参数被拦截：工具调用没通过参数校验（模型写错了参数），占全部工具调用的比例。",
        "- 权限拒绝：工具层拒绝越权调用。安全用例里这是预期行为，不算系统错误。",
        "- 接口异常：客户端自动重试 2 次，仍失败会中断整次运行；表中各次运行都没有中断，所以没有单列。",
        "",
        "| 配置 | 空答案 | 截断 | 未正常结束 | 参数被拦截 / 工具调用 | 权限拒绝 | 护栏退回 | 最终仍有严重引用问题 |",
        "|---|---|---|---|---|---|---|---|",
        *err_table,
        "",
        "## 评委成本",
        "",
        "评委只在离线评测时调用，不在用户请求路径上。",
        "",
        "| 被评的配置 | 题数 | 解析失败 | 每题输入 / 输出 token | P50 秒 | 美元 / 千题 |",
        "|---|---|---|---|---|---|",
        *judge_table,
    ]
    if notes:
        lines += ["", "## 未纳入", "", *[f"- {x}" for x in notes]]
    out = REPORTS / f"{args.name}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
