"""权限、隔离与脱敏测试。不依赖向量服务：查询向量用离线算好的题目向量代替。"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from access import DENIED, INDUSTRY_DOC, PRODUCT_DOC, load_principal  # noqa: E402
from privacy import redact  # noqa: E402

EMB = ROOT / "data" / "processed" / "emb" / "bge-m3_index_text"
pytestmark = pytest.mark.skipif(not (EMB / "chunks.npy").exists(), reason="需要 chunks 和向量")


@pytest.fixture(scope="module")
def corpus():
    from agent import Corpus

    c = Corpus()
    qv = np.load(EMB / "questions.npy")
    c.embed = lambda text: qv[1]  # q002：泰康等待期问题的向量，内容与测试无关，只要固定
    return c


def episode(corpus, pid):
    from agent import TOOLSETS, Episode

    return Episode(corpus, "", 0, TOOLSETS["full"], load_principal(pid))


def cited_docs(ep):
    by_id = {c["chunk_id"]: c for c in ep.corpus.chunks}
    return [by_id[b["chunk_ids"][0]]["doc_id"] for b in ep.blocks if b["chunk_ids"]]


def test_staff_search_is_prefiltered(corpus):
    """泰康员工检索：5 条结果全部来自泰康条款或行业规范，而且凑满 5 条（说明是先过滤再排序）。"""
    ep = episode(corpus, "S_TK")
    ep.search("等待期多长")
    docs = cited_docs(ep)
    assert len(docs) == 5
    assert set(docs) <= {PRODUCT_DOC["泰康惠嘉保"], INDUSTRY_DOC}


def test_staff_cannot_search_or_read_other_tenant(corpus):
    ep = episode(corpus, "S_TK")
    assert ep.search("等待期", products=["太保阿基米德"]).startswith("错误：无权")
    assert ep.read_clause("太保阿基米德", "2.4.2").startswith("错误：无权")
    assert not ep.blocks and len(ep.denials) == 2


def test_customer_can_search_all_public_clauses(corpus):
    ep = episode(corpus, "U2")
    ep.search("等待期", products=["太保阿基米德"])
    assert set(cited_docs(ep)) == {PRODUCT_DOC["太保阿基米德"]}


def test_customer_policy_isolation_and_no_existence_leak(corpus):
    ep = episode(corpus, "U2")
    other = ep.calc("policy_lookup", {"policy_id": "P001"})  # 存在，但属于 U1
    missing = ep.calc("policy_lookup", {"policy_id": "P999"})  # 不存在
    assert other == missing == f"错误：{DENIED}"
    assert "P002" in ep.calc("policy_lookup", {"policy_id": "P002"})
    denied_calc = ep.calc("benefit_calc", {"policy_id": "P003", "level": "轻症", "diagnosis_date": "2027-01-10"})
    assert denied_calc == f"错误：{DENIED}"


def test_staff_policy_scope_by_tenant(corpus):
    ep = episode(corpus, "S_CL")
    ok = ep.calc("benefit_calc", {"policy_id": "P003", "level": "轻症", "diagnosis_date": "2027-01-10"})
    assert '"amount": 80000' in ok
    assert ep.calc("policy_lookup", {"policy_id": "P001"}) == f"错误：{DENIED}"


def test_redact_pii_keeps_business_ids():
    text, hits = redact("我身份证110101199003071234，手机13812345678，卡号6222021234567890123，邮箱a.b@example.com，保单P001，2026-09-20确诊，保额500000")
    assert "110101199003071234" not in text and "13812345678" not in text and "6222021234567890123" not in text
    assert "a.b@example.com" not in text
    assert sorted(hits) == sorted(["身份证号", "银行卡号", "手机号", "邮箱"])
    assert "P001" in text and "2026-09-20" in text and "500000" in text
