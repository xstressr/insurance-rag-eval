"""个人信息脱敏：用户输入在发给模型、写进日志之前，先替换身份证号、手机号、银行卡号、邮箱。

只处理格式明确的标识符，不做姓名、地址识别（误伤多，需要 NER）。保单号（P001）、日期、金额不受影响。
"""

from __future__ import annotations

import re

PATTERNS = [
    ("身份证号", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("银行卡号", re.compile(r"(?<!\d)\d{16,19}(?!\d)")),
    ("手机号", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("邮箱", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
]


def redact(text: str) -> tuple[str, list[str]]:
    """返回 (脱敏后的文本, 命中的类型列表)。身份证号先于银行卡号匹配，避免 18 位身份证被当成卡号。"""
    hits = []
    for name, pat in PATTERNS:
        text, n = pat.subn(f"[{name}已脱敏]", text)
        hits += [name] * n
    return text, hits
