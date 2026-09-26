"""计算规则的测试：规则引用的条款原文确实存在；工具结果与手算的标准答案一致。"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rules import RULES, ToolError, benefit_calc, date_calc  # noqa: E402

CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
TASKS = ROOT / "dataset" / "tool_tasks_v1.jsonl"


def _clause_text(doc, part, clause_id):
    chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
    return "".join(c["text"] for c in chunks if (c["doc_id"], c["part"], c["clause_id"]) == (doc, part, clause_id))


@pytest.mark.skipif(not CHUNKS.exists(), reason="先运行 src/ingest.py")
@pytest.mark.parametrize("key", list(RULES))
def test_rule_evidence_exists_in_clause(key):
    rule = RULES[key]
    text = _clause_text(*rule.clause)
    assert text, f"找不到条款 {rule.clause}"
    for phrase in rule.evidence:
        assert phrase in text, f"{key} 的依据“{phrase}”不在 {rule.clause} 原文中"


@pytest.mark.skipif(not CHUNKS.exists(), reason="先运行 src/ingest.py")
def test_taikang_waiting_period_clause():
    text = _clause_text("taikang_huijiabao_2026", "", "1.3")
    assert "90 日的等待期" in text and "无息退还" in text


def test_tasks_match_hand_derivation():
    """标准答案是人工推算的，计算工具必须得出同样的结果。"""
    for t in map(json.loads, TASKS.open(encoding="utf-8")):
        if t["id"] == "t07":
            assert date_calc("2027-05-10", 60, count_start=False)["last_day"] == t["expected"]
            continue
        q = t["question"]
        pid = q[q.index("P") : q.index("P") + 4]
        level = "重疾" if "重疾" in q else ("中症" if "中症" in q else "轻症")
        diag = "-".join(f"{int(x):02d}" for x in __import__("re").findall(r"(\d{4})年(\d{1,2})月(\d{1,2})日", q)[0])
        got = benefit_calc(pid, level, diag, accident="意外）" in q and "不是意外" not in q and "非意外" not in q)
        assert got["amount"] == t["expected"], (t["id"], got)


def test_tool_errors():
    with pytest.raises(ToolError):
        benefit_calc("P999", "轻症", "2026-01-01")
    with pytest.raises(ToolError):
        benefit_calc("P001", "特疾", "2026-01-01")
    with pytest.raises(ToolError):
        date_calc("2026/01/01", 10)
