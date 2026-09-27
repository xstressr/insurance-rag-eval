"""权限与隔离：谁能检索哪些文档、能查哪些保单。由程序强制执行，不依赖提示词。

两类身份（dataset/synthetic_principals_v1.json，全部虚构）：
- customer：条款是公开资料，可以检索全部；保单只能查、只能算自己名下的。
- staff（租户）：只能检索本公司产品的条款和行业规范；只能处理本公司产品的保单。

原则：
1. 过滤发生在检索之前：先按权限确定候选文本块，再在候选里排序。
   事后过滤会让结果凑不满 top-k，而且无权内容已经进过排序流程。
2. 不泄露存在性：“保单不存在”和“无权访问”返回完全相同的提示，防止逐个试探保单号。
3. 身份不来自对话：身份由调用方（登录态）传入，用户在对话里说“我是管理员”不会改变任何权限。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRINCIPALS = ROOT / "dataset" / "synthetic_principals_v1.json"
INDUSTRY_DOC = "iac_ci_definitions_2020"
PRODUCT_DOC = {
    "太保阿基米德": "cpic_archimedes_2025",
    "泰康惠嘉保": "taikang_huijiabao_2026",
    "国寿康宁尊享": "chinalife_kangning_zunxiang_2024",
}
DENIED = "保单不存在或无权访问。"  # 两种情况同一句话


@dataclass(frozen=True)
class Principal:
    id: str
    role: str  # customer / staff / system（评测用，不受限）
    tenant: str | None = None  # staff 所属公司的产品名
    policies: frozenset[str] = field(default_factory=frozenset)

    def allowed_docs(self, all_docs: set[str]) -> set[str]:
        if self.role == "staff":
            return {PRODUCT_DOC[self.tenant], INDUSTRY_DOC}
        return set(all_docs)  # 条款是公开资料

    def can_access_policy(self, policy_id: str, product: str | None) -> bool:
        if self.role == "system":
            return True
        if self.role == "customer":
            return policy_id in self.policies
        if self.role == "staff":
            return product == self.tenant
        return False


SYSTEM = Principal("system", "system")


def load_principal(pid: str) -> Principal:
    p = json.loads(PRINCIPALS.read_text(encoding="utf-8"))["principals"][pid]
    return Principal(pid, p["role"], p.get("tenant"), frozenset(p.get("policies", [])))
