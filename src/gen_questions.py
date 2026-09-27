"""blind_v3：让没见过条款的模型出题，再按事先写定的规则机械筛选。规则见 dataset/blind_v3_protocol.md。

    uv run python src/gen_questions.py generate   # 调用出题模型，写 dataset/blind_v3_candidates.jsonl
    uv run python src/gen_questions.py select     # 去重 + 分层抽样，写 dataset/golden_blind_v3.jsonl

去重需要本地向量服务（src/embed_server.py）。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
import uuid

import numpy as np

from agent import EMBED_URL
from answer import LLM
from evaluate import ROOT, load_jsonl

DATASET = ROOT / "dataset"
CANDIDATES = DATASET / "blind_v3_candidates.jsonl"
OUT = DATASET / "golden_blind_v3.jsonl"
WRITERS = ["kimi-k3", "qwen3.8-max", "muse-spark-1.3-contributor"]  # 顺序也是去重和补足的顺序
QUOTA = {"kimi-k3": 7, "qwen3.8-max": 7, "muse-spark-1.3-contributor": 6}
PER_WRITER, SEED, SIM = 15, 20260927, 0.90
EXISTING = ["golden_v2.jsonl", "golden_blind_v1.jsonl", "golden_blind_v2.jsonl", "tool_tasks_v1.jsonl", "security_cases_v2.jsonl"]

WRITER_PROMPT = """你在帮忙测试一个“保险条款问答助手”。助手能查阅下面三款重大疾病保险的条款原文，以及中国保险行业协会的《重大疾病保险的疾病定义使用规范（2020年修订版）》：

1. 太保阿基米德（2025）重大疾病保险（互联网）——中国太平洋人寿
2. 泰康惠嘉保2026重大疾病保险——泰康人寿
3. 国寿康宁尊享重大疾病保险（2024版）——中国人寿

请以真实用户的身份，写 {n} 个会问这个助手的问题。要求：
- 口语化，像普通人在手机上打字，不要写成考试题；
- 场景分散：买之前比较和咨询、买了以后交费/退保/借款/改信息、生病或出险后理赔、家属或受益人的问题、健康告知；
- 大部分问题只问其中一款产品，少数问题对比两三款，少数问题问疾病定义本身；
- 可以有条款里未必写到的问题，真实用户就是这样问的；
- 不要编造你不确定的条款细节，也不要给出答案；
- 不要包含任何真实的个人信息。

只输出一个 JSON 数组，每个元素形如 {{"question": "...", "scenario": "..."}}，不要输出其他内容。"""


def embed(texts: list[str]) -> np.ndarray:
    out = []
    for i in range(0, len(texts), 32):
        req = urllib.request.Request(
            EMBED_URL, data=json.dumps({"texts": texts[i : i + 32]}).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            out += json.load(resp)["vectors"]
    return np.array(out, dtype=np.float32)


def parse_array(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.S)
    items = json.loads(m.group(0)) if m else []
    return [x for x in items if isinstance(x, dict) and isinstance(x.get("question"), str)]


def generate() -> None:
    rows = []
    for w in WRITERS:
        llm = LLM(None, model=w)
        session = str(uuid.uuid5(uuid.NAMESPACE_URL, f"insurance-rag-eval/blind_v3/{w}"))
        resp = llm.chat("你是一名普通的保险消费者。", WRITER_PROMPT.format(n=PER_WRITER), session)
        items = parse_array(resp["content"])
        print(f"{w}: {len(items)} 题，cached={resp['cached']}，{resp['usage']}")
        rows += [{"writer": w, "rank": i, "question": x["question"].strip(), "scenario": x.get("scenario", "")} for i, x in enumerate(items)]
    with CANDIDATES.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def select() -> None:
    cands = load_jsonl(CANDIDATES)
    existing = [q["question"] for name in EXISTING for q in load_jsonl(DATASET / name)]
    ok = [c for c in cands if 6 <= len(c["question"]) <= 120 and re.search(r"[一-鿿]", c["question"])]
    vc, ve = embed([c["question"] for c in ok]), embed(existing)
    kept, kept_vecs = [], []
    for c, v in sorted(zip(ok, vc), key=lambda cv: (WRITERS.index(cv[0]["writer"]), cv[0]["rank"])):
        c["max_sim_existing"] = round(float((ve @ v).max()), 3)
        if c["max_sim_existing"] >= SIM:
            c["dropped"] = "与已有题目重复"
            continue
        if kept_vecs and float((np.array(kept_vecs) @ v).max()) >= SIM:
            c["dropped"] = "与其他候选重复"
            continue
        kept.append(c)
        kept_vecs.append(v)

    rng = random.Random(SEED)
    pools = {w: [c for c in kept if c["writer"] == w] for w in WRITERS}
    for w in WRITERS:
        rng.shuffle(pools[w])
    picked = []
    for w in WRITERS:
        picked += pools[w][: QUOTA[w]]
        pools[w] = pools[w][QUOTA[w] :]
    for w in WRITERS:  # 某个模型不够配额时，按顺序从剩余候选补足
        while len(picked) < sum(QUOTA.values()) and pools[w]:
            picked.append(pools[w].pop(0))

    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for i, c in enumerate(picked, 1):
            f.write(json.dumps({
                "id": f"d{i:02d}", "question": c["question"], "writer": c["writer"], "scenario": c["scenario"],
                "status": "frozen_unlabeled", "author": c["writer"], "written_at": "2026-09-27",
                "blindness": "出题模型没见过条款、已有题目和系统输出；筛选按 dataset/blind_v3_protocol.md 机械执行",
            }, ensure_ascii=False) + "\n")
    print(f"候选 {len(cands)}，格式合格 {len(ok)}，去重后 {len(kept)}，抽中 {len(picked)}")
    for w in WRITERS:
        print(f"  {w}: 候选 {sum(c['writer'] == w for c in cands)}，保留 {sum(c['writer'] == w for c in kept)}，抽中 {sum(c['writer'] == w for c in picked)}")
    with CANDIDATES.open("w", encoding="utf-8", newline="\n") as f:  # 把丢弃原因和相似度写回候选全集，方便审计
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["generate", "select"])
    {"generate": generate, "select": select}[ap.parse_args().stage]()
