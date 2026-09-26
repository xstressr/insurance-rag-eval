"""确定性计算工具：保单查询、理赔金额计算、日期推算。

规则由 AI 从条款原文编码（2026-09-26），每条都带出处和原文关键句；
tests/test_rules.py 会检查这些关键句真实出现在对应条款里，防止规则写错。

范围（有意收窄，只覆盖能从条款确定算出的部分）：
- 重疾、中症、轻症三个等级；身故、豁免、关爱金、特定疾病额外给付等暂不计算；
- 现金价值无法从条款得到（在保单上载明），重疾“三者取大”只比较基本保额和已交保费，并在结果里说明。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POLICIES = ROOT / "dataset" / "synthetic_policies_v1.json"

DOC = {
    "太保阿基米德": "cpic_archimedes_2025",
    "泰康惠嘉保": "taikang_huijiabao_2026",
    "国寿康宁尊享": "chinalife_kangning_zunxiang_2024",
}


@dataclass(frozen=True)
class Rule:
    ratio: float | None  # 按基本保额的比例；None 表示“基本保额与已交保费取大”（另有现金价值，条款外）
    max_times: int
    waiting_days: int  # 非意外原因的观察期
    within_waiting: str  # "refund_terminate"：退还已交保费、合同终止；"no_pay"：不赔、合同继续有效
    option: str | None  # 需要投保的可选责任
    clause: tuple[str, str, str]  # (doc_id, part, clause_id)
    evidence: tuple[str, ...]  # 条款原文关键句，测试会逐条核对


RULES: dict[tuple[str, str], Rule] = {
    ("太保阿基米德", "重疾"): Rule(
        None, 1, 90, "refund_terminate", None, ("cpic_archimedes_2025", "", "2.4.1"),
        ("按以下三项中金额较大者给付重大疾病保险金", "90 日内因意外伤害以外的原因被确诊初次发生本合同约定的重大疾病，我们按您根据本合同约定已支付的保险费总额给付重大疾病保险金，本合同终止"),
    ),
    ("太保阿基米德", "中症"): Rule(
        0.6, 3, 90, "no_pay", "可选保障1", ("cpic_archimedes_2025", "", "2.4.2"),
        ("按本合同基本保险金额的60%给付中症疾病保险金", "中症疾病保险金累计给付以三次为限", "90 日内因意外伤害以外的原因被确诊初次发生本合同约定的中症疾病，我们不承担给付中症疾病保险金的责任，本合同继续有效"),
    ),
    ("太保阿基米德", "轻症"): Rule(
        0.3, 4, 90, "no_pay", "可选保障1", ("cpic_archimedes_2025", "", "2.4.2"),
        ("按本合同基本保险金额的30%给付轻症疾病保险金", "轻症疾病保险金累计给付以四次为限", "90 日内因意外伤害以外的原因被确诊初次发生本合同约定的轻症疾病，我们不承担给付轻症疾病保险金的责任，本合同继续有效"),
    ),
    ("泰康惠嘉保", "重疾"): Rule(
        1.0, 1, 90, "refund_terminate", None, ("taikang_huijiabao_2026", "", "1.4"),
        ("我们将按本合同的基本保险金额向疾病保险金受益人给付重大疾病保险金，本合同终止",),
    ),
    ("国寿康宁尊享", "重疾"): Rule(
        None, 1, 180, "refund_terminate", None, ("chinalife_kangning_zunxiang_2024", "利益条款", "第九条"),
        ("一百八十日内，因首次发生并经确诊的疾病导致被保险人初次发生并经专科医生明确诊断患本合同所指的重大疾病（无论一种或多种），本合同终止，本公司按本合同所交保险费（不计利息）给付重大疾病保险金", "下列三者的较大值给付重大疾病保险金"),
    ),
    ("国寿康宁尊享", "轻症"): Rule(
        0.2, 6, 180, "refund_terminate", "可选保险责任一", ("chinalife_kangning_zunxiang_2024", "利益条款", "第九条"),
        ("本公司按本合同基本保险金额的20%给付轻度疾病保险金", "本合同的轻度疾病保险金累计给付以六次为限", "本合同终止，本公司按本合同所交保险费（不计利息）给付轻度疾病保险金"),
    ),
}
# 条款依据：泰康惠嘉保 1.3 等待期 90 日、等待期内退还已交保费并终止（1.4 的举例表格也印证）；
# 泰康无中症、轻症责任；国寿无“中症”，其“轻度疾病”在这里按“轻症”处理。


class ToolError(ValueError):
    """参数合法但业务上无法处理（保单不存在、日期格式错误等），返回给模型让它改正。"""


def load_policies() -> dict[str, dict]:
    return json.loads(POLICIES.read_text(encoding="utf-8"))["policies"]


def _date(s: str, name: str) -> date:
    try:
        return date.fromisoformat(s)
    except (TypeError, ValueError):
        raise ToolError(f"{name} 必须是 YYYY-MM-DD 格式的日期，收到“{s}”") from None


def policy_lookup(policy_id: str) -> dict:
    policies = load_policies()
    if policy_id not in policies:
        raise ToolError(f"保单 {policy_id} 不存在。")
    return {"policy_id": policy_id, **policies[policy_id]}


def benefit_calc(policy_id: str, level: str, diagnosis_date: str, accident: bool = False) -> dict:
    """按条款规则计算一次确诊应赔多少。返回金额、结论和计算过程，过程里写明依据条款。"""
    p = policy_lookup(policy_id)
    product = p["product"]
    if level not in ("重疾", "中症", "轻症"):
        raise ToolError(f"level 只能是 重疾 / 中症 / 轻症，收到“{level}”")
    rule = RULES.get((product, level))
    steps = [f"保单 {policy_id}：{product}，基本保额 {p['sum_insured']} 元，生效日 {p['effective_date']}。"]
    if rule is None:
        return {"amount": 0, "outcome": "不在保障范围", "steps": steps + [f"{product} 没有{level}责任。"]}
    ref = f"依据：{rule.clause[1] + ' ' if rule.clause[1] else ''}{rule.clause[2]}"
    if rule.option and rule.option not in p["options"]:
        return {"amount": 0, "outcome": "未投保对应可选责任", "steps": steps + [f"{level}属于{rule.option}，该保单未投保。{ref}"]}
    used = p["claims"].get(level, 0)
    if used >= rule.max_times:
        return {"amount": 0, "outcome": "已达给付次数上限", "steps": steps + [f"{level}累计给付以 {rule.max_times} 次为限，已赔 {used} 次。{ref}"]}

    diag = _date(diagnosis_date, "diagnosis_date")
    eff = _date(p["effective_date"], "effective_date")
    if diag < eff:
        raise ToolError("确诊日期早于保单生效日，不在保险期间内。")
    days = (diag - eff).days
    steps.append(f"确诊日 {diagnosis_date}，距生效日 {days} 日；{'意外导致，不受观察期限制' if accident else f'非意外，观察期 {rule.waiting_days} 日'}。")
    if not accident and days < rule.waiting_days:
        if rule.within_waiting == "refund_terminate":
            steps.append(f"在观察期内：合同终止，按已交保费 {p['premiums_paid']} 元给付（退还）。{ref}")
            return {"amount": p["premiums_paid"], "outcome": "观察期内确诊：退还已交保费，合同终止", "steps": steps}
        steps.append(f"在观察期内：不承担{level}保险金责任，合同继续有效。{ref}")
        return {"amount": 0, "outcome": "观察期内确诊：不赔，合同继续有效", "steps": steps}

    if rule.ratio is None:
        amount = max(p["sum_insured"], p["premiums_paid"])
        steps.append(f"取基本保额 {p['sum_insured']} 与已交保费 {p['premiums_paid']} 的较大者 = {amount} 元；条款还要求与确诊时的现金价值比较，现金价值以保单载明为准，此处未计入。{ref}")
        return {"amount": amount, "outcome": "给付（不低于此金额，现金价值更高时按现金价值）", "steps": steps}
    amount = round(p["sum_insured"] * rule.ratio)
    steps.append(f"{level}按基本保额的 {rule.ratio:.0%} 给付：{p['sum_insured']} × {rule.ratio:.0%} = {amount} 元；这是第 {used + 1} 次，上限 {rule.max_times} 次。{ref}")
    return {"amount": amount, "outcome": "给付", "steps": steps}


def date_calc(start_date: str, days: int, count_start: bool = True) -> dict:
    """从 start_date 起算 days 日，返回最后一天。
    count_start=True：start_date 算第 1 日（“自某日起 N 日”）；False：从次日起算（“自次日起 N 日”）。"""
    start = _date(start_date, "start_date")
    if not isinstance(days, int) or days <= 0:
        raise ToolError("days 必须是正整数。")
    first = start if count_start else start + timedelta(days=1)
    last = first + timedelta(days=days - 1)
    return {"first_day": first.isoformat(), "last_day": last.isoformat(), "days": days}
