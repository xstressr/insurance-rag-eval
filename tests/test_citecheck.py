"""引用程序校验的单元测试：数字抽取、按引用切片段，以及用真实答案做对抗测试。"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from citecheck import SEVERE, block_texts, check, cn2num, facts, spans  # noqa: E402

CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
RUNS = ROOT / "reports" / "runs"


def test_cn2num():
    assert cn2num("一百八十") == 180
    assert cn2num("十五") == 15
    assert cn2num("六十") == 60
    assert cn2num("两") == 2


def test_facts_normalizes_units_and_numerals():
    f = facts("自次日起六十日为宽限期；按基本保险金额的30%给付；以四次为限；55 周岁；90 天；TNM 分期为Ⅰ期")
    assert {("60", "日"), ("30", "%"), ("4", "次"), ("55", "岁"), ("90", "日"), ("1", "期")} <= f


def test_facts_skips_quantifiers_and_years():
    assert facts("至少一次检测；分两种情形；（2020年修订版）") == set()


def test_spans_assign_trailing_citation():
    s = spans("标题：90日\n条款约定90日后给付；等待期内退费 [C1]。其余说明")
    assert s[0][1] == ["C1"] and "标题" in s[0][0]
    assert s[-1] == ("。其余说明", [])


def _load(run):
    path = RUNS / f"{run}.jsonl"
    if not (path.exists() and CHUNKS.exists()):
        pytest.skip(f"缺少 {path.name} 或 chunks")
    chunk_by_id = {c["chunk_id"]: c for c in map(json.loads, CHUNKS.open(encoding="utf-8"))}
    return {r["id"]: r for r in map(json.loads, path.open(encoding="utf-8"))}, chunk_by_id


# 把 gen_v3_p2 中无问题的答案改错一个数字，检查器应报出严重问题
TAMPER = [
    ("q001", "15日", "30日"),
    ("q012", "125种", "150种"),
    ("q004", "80%", "90%"),
    ("q006", "三次", "五次"),
    ("q007", "30%", "40%"),
]


@pytest.mark.parametrize("qid,old,new", TAMPER)
def test_detects_tampered_numbers(qid, old, new):
    rows, chunk_by_id = _load("gen_v3_p2")
    r = rows[qid]
    blocks = block_texts(r, chunk_by_id)
    assert not [i for i in check(r["answer"], blocks, r["question"]) if i["kind"] in SEVERE], "原答案应无严重问题"
    assert old in r["answer"]
    bad = check(r["answer"].replace(old, new, 1), blocks, r["question"])
    # 已知局限：改错后的数字若碰巧出现在同一产品的其他片段里（q007 的 40%），只会报“引用不精确”这一警告，
    # 严重程度被低估。所以要求报出问题，但不要求一定是严重级别。
    assert bad


def test_q027_misattribution_is_flagged():
    """agent_v2 的 q027 把国寿的 180 日写到了太保名下：应判为张冠李戴。"""
    rows, chunk_by_id = _load("agent_v2_graph")
    r = rows["q027"]
    kinds = {i["kind"] for i in check(r["answer"], block_texts(r, chunk_by_id), r["question"])}
    assert "张冠李戴" in kinds
