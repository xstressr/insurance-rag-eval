"""引用的程序校验：答案里每句话中带单位的数字，必须出现在这句话自己引用的片段里。

    uv run python src/citecheck.py agent_v2_graph agent_v2_graph_blind

按“引用片段”检查：模型常把几个分句写在一起、只在末尾标一次引用（“……；……；……[C1]。”），
所以从上一个引用标记到下一个引用标记之间的文字，都由后面那个引用负责；最后一个引用之后的文字才算未引用。
标题行（“**1. 太保：90日**”）没有引用，会并入下一个片段，由下一个引用负责。

为什么只在“这段引用的片段”里找，而不是在全部上下文里找：
agent_v2 的 q027 把国寿的“180 日”写到了太保名下。180 日在上下文里真实存在（国寿片段），
只在“这句话引用的太保片段”里找不到。全局查找会放过这类张冠李戴。

只查带单位的数字（天数、比例、次数、年龄、金额……），不查条款号、页码这类编号。
这是评委之外的第二道防线：便宜、确定、可复现，但只能覆盖数字类事实。
"""

from __future__ import annotations

import json
import re
import sys

from evaluate import CHUNKS, REPORTS, load_jsonl

UNITS = ["个保单年度", "个月", "周岁", "万元", "小时", "日", "天", "%", "％", "次", "岁", "种", "年", "元", "倍", "组", "级", "期"]
UNIT_NORM = {"天": "日", "％": "%", "周岁": "岁"}
CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNIT = {"十": 10, "百": 100, "千": 1000}
_UNIT_RE = "|".join(sorted(map(re.escape, UNITS), key=len, reverse=True))
FACT_RE = re.compile(rf"(\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千]+)\s*({_UNIT_RE})")


def cn2num(s: str) -> int | None:
    """中文数字转整数，支持到千位：十五 → 15，一百八十 → 180，两 → 2。"""
    if not s:
        return None
    total, cur = 0, 0
    for ch in s:
        if ch in CN_DIGIT:
            cur = CN_DIGIT[ch]
        elif ch in CN_UNIT:
            total += (cur or 1) * CN_UNIT[ch]
            cur = 0
        else:
            return None
    return total + cur


ROMAN = str.maketrans({"Ⅰ": "1", "Ⅱ": "2", "Ⅲ": "3", "Ⅳ": "4", "Ⅴ": "5"})
# 日期不是数字事实：“2026年5月10日”里的“10日”会被误当成天数。抽取前先整体去掉。
DATE_RE = re.compile(r"\d{4}-\d{1,2}-\d{1,2}|(?:\d{4}\s*年\s*)?\d{1,2}\s*月\s*\d{1,2}\s*日|\d{4}\s*年(?!\s*内|\s*以)")
BARE_NUM = re.compile(r"\d+(?:\.\d+)?")


def facts(text: str) -> set[tuple[str, str]]:
    """抽取 (数值, 单位)。数值统一成阿拉伯数字字符串，单位做同义归一。

    不算事实的：“一次”（量词）；小于 10 的“种”（多是模型自己数的）；1900 年以后的年份（多是文件名里的年份）。
    """
    out = set()
    for num, unit in FACT_RE.findall(DATE_RE.sub(" ", text.translate(ROMAN))):
        if not num[0].isdigit():
            n = cn2num(num)
            if n is None:
                continue
            num = str(n)
        elif "." in num:
            num = num.rstrip("0").rstrip(".")
        unit = UNIT_NORM.get(unit, unit)
        # “种”小于 10 多是模型自己数的（“分两种情形”），清单总数（125 种、40 种）才是条款事实
        if (num == "1" and unit == "次") or (unit == "种" and float(num) < 10) or (unit == "年" and float(num) >= 1900):
            continue
        out.add((num, unit))
    return out


def split_sentences(answer: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?<=[。；！？])|\n", answer) if p.strip()]


def spans(answer: str) -> list[tuple[str, list[str]]]:
    """按引用标记切片段：返回 [(文字, 负责它的引用编号)]。连续的 [C1][C3] 算一组；末尾没有引用的文字编号为空。"""
    out, pos = [], 0
    for m in re.finditer(r"(?:\[C\d+\])+", answer):
        out.append((answer[pos : m.start()], re.findall(r"C\d+", m.group(0))))
        pos = m.end()
    out.append((answer[pos:], []))
    return [(t, c) for t, c in out if t.strip()]


SEVERE = ("张冠李戴", "上下文中不存在", "未引用")


def check(answer: str, blocks: dict[str, tuple[str, set[str]]], question: str = "") -> list[dict]:
    """blocks: {Cn: (片段文字, 所属文档集合)}。返回问题列表，每项含片段文字、数字事实和问题类型：
    - 未引用：带数字的文字后面没有任何引用；
    - 引用不精确（警告，需复核）：数字不在这段引用的片段里，但在同一产品的其他片段里。多是引用标少了，
      但也可能是数字写错、碰巧与同产品别处的数字相同（如把 30% 写成 40%），严重程度会被低估；
    - 张冠李戴：数字只出现在另一个产品的片段里（如把国寿的 180 日写成太保的）；
    - 上下文中不存在：模型看过的所有片段里都没有这个数字（编造或自行推算）。
    问题本身出现的数字（如“30岁买……”）视为复述，不检查。"""
    block_text = {c: t for c, (t, _) in blocks.items()}
    block_facts = {c: facts(t) for c, t in block_text.items()}
    # 计算工具的结果是 JSON（"amount": 0、"days": 60），数字不带单位：只要数值相同就算支持
    tool_nums = {c: {n.rstrip("0").rstrip(".") if "." in n else n for n in BARE_NUM.findall(DATE_RE.sub(" ", t))}
                 for c, (t, docs) in blocks.items() if docs == {"tool"}}
    asked = facts(question)
    issues = []
    for text, cids in spans(answer):
        fs = facts(text) - asked
        if not fs:
            continue
        if not cids:
            issues.append({"sentence": text.strip(), "facts": sorted(fs), "kind": "未引用"})
            continue
        support = set().union(*(block_facts.get(c, set()) for c in cids))
        cited_nums = set().union(*(tool_nums.get(c, set()) for c in cids))
        cited_docs = set().union(*(blocks[c][1] for c in cids if c in blocks))
        for f in sorted(f for f in fs - support if f[0] not in cited_nums):
            holders = [c for c, bf in block_facts.items() if f in bf] + [c for c, ns in tool_nums.items() if f[0] in ns]
            if not holders:
                kind = "上下文中不存在"
            elif any(blocks[c][1] & cited_docs for c in holders):
                kind = "引用不精确"
            else:
                kind = "张冠李戴"
            issues.append({"sentence": text.strip(), "facts": [f], "kind": kind, "cited": cids})
    return issues


def block_texts(row: dict, chunk_by_id: dict[str, dict]) -> dict[str, tuple[str, set[str]]]:
    return {
        b["cid"]: ("".join(chunk_by_id[x]["text"] for x in b["chunk_ids"]), {chunk_by_id[x]["doc_id"] for x in b["chunk_ids"]})
        for b in row["context"]
    }


def main() -> None:
    runs = sys.argv[1:]
    chunk_by_id = {c["chunk_id"]: c for c in load_jsonl(CHUNKS)}
    for run in runs:
        rows = load_jsonl(REPORTS / "runs" / f"{run}.jsonl")
        judge_path = REPORTS / "runs" / f"judge_{run}.jsonl"
        judge = {r["id"]: r for r in load_jsonl(judge_path)} if judge_path.exists() else {}
        n_facts = flagged = 0
        kinds: dict[str, int] = {}
        agree = {"都判有问题": [], "只有程序判": [], "只有评委判": []}
        for r in rows:
            issues = check(r["answer"], block_texts(r, chunk_by_id), r["question"])
            for it in issues:
                kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
            severe = [it for it in issues if it["kind"] in SEVERE]
            n_facts += sum(len(facts(t)) for t, _ in spans(r["answer"]))
            flagged += bool(severe)
            j_bad = judge.get(r["id"], {}).get("faithful") is False
            if severe and j_bad:
                agree["都判有问题"].append(r["id"])
            elif severe:
                agree["只有程序判"].append(r["id"])
            elif j_bad:
                agree["只有评委判"].append(r["id"])
            for it in issues:
                print(f"  [{run}] {r['id']} {it['kind']} {it['facts']} | {it['sentence'][-90:]}")
        print(f"{run}: 数字事实 {n_facts} 个，问题 {kinds}，有严重问题的答案 {flagged}/{len(rows)}；与评委对照 {agree}")


if __name__ == "__main__":
    main()
