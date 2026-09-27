"""拆子问题的纯函数：解析拆分结果、合并多路检索排序。不需要数据和模型。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from answer import merge_seeds, parse_subquestions  # noqa: E402


def test_parse_subquestions_tolerates_wrapping_and_caps_at_four():
    text = '好的：\n```json\n{"subquestions": ["保险期间", " ", "宽限期", 3, "现金价值", "保单贷款", "受益人"]}\n```'
    assert parse_subquestions(text) == ["保险期间", "宽限期", "现金价值", "保单贷款"]


def test_parse_subquestions_bad_output_means_no_split():
    assert parse_subquestions("无法拆分") == []
    assert parse_subquestions('{"subquestions": "宽限期"}') == []
    assert parse_subquestions('{"queries": ["宽限期"]}') == []


def test_merge_seeds_round_robin_dedup_and_per_list_cap():
    orig = [1, 2, 3, 4, 5]
    sub_a = [2, 7, 8, 9]
    sub_b = [7, 10, 11]
    # 每轮按原问题、检索词 a、检索词 b 的顺序各取下一名；重复的跳过（也占掉这一名）；检索词只取前 2 名
    assert merge_seeds([orig, sub_a, sub_b], [5, 2, 2]) == [1, 2, 7, 10, 3, 4, 5]


def test_merge_seeds_single_list_is_prefix():
    assert merge_seeds([[4, 3, 2, 1]], [3]) == [4, 3, 2]
