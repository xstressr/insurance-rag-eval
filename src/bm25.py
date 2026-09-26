"""中文分词与手写 BM25。

BM25 给“问题-文本块”打分：
    score = Σ_词 IDF(词) × tf × (k1 + 1) / (tf + k1 × (1 − b + b × 块长 / 平均块长))
- tf：词在块里出现的次数；k1 控制次数收益的饱和速度。
- IDF：含这个词的块越少，这个词越有区分度。
- b：长度惩罚力度，防止长块因为词多而占便宜。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import jieba

TERMS_FILE = Path(__file__).resolve().parent / "resources" / "insurance_terms.txt"

# 只去掉虚词和疑问词；“本合同”这类高频实词交给 IDF 自动降权，不手工删。
STOPWORDS = set(
    "的 了 是 在 和 与 或 及 等 之 其 为 以 对 将 被 由 于 也 就 都 而 并 且 "
    "吗 呢 吧 啊 么 什么 怎么 怎样 多少 多久 多长 哪 哪些 哪个 几 如何 可以 能 会 要 还 呀".split()
)
HAS_WORD_CHAR = re.compile(r"[一-鿿A-Za-z0-9]")

_dict_loaded = False


def _load_terms() -> None:
    global _dict_loaded
    if _dict_loaded:
        return
    jieba.setLogLevel(60)  # 关掉 jieba 首次加载时的提示输出
    for line in TERMS_FILE.read_text(encoding="utf-8").splitlines():
        word = line.strip()
        if word and not word.startswith("#"):
            jieba.add_word(word, freq=100_000)  # 高词频保证整词优先
    _dict_loaded = True


def tokenize(text: str) -> list[str]:
    """jieba 精确模式分词，去掉标点、空白和停用词，英文转小写。"""
    _load_terms()
    tokens = []
    for tok in jieba.lcut(text):
        tok = tok.strip().lower()
        if tok and HAS_WORD_CHAR.search(tok) and tok not in STOPWORDS:
            tokens.append(tok)
    return tokens


class BM25:
    """Okapi BM25，带倒排表。

    idf_variant:
    - "lucene"（默认）：log(1 + (N − n + 0.5) / (n + 0.5))，恒为正，Elasticsearch / Lucene 的做法。
    - "okapi"：log((N − n + 0.5) / (n + 0.5))，负值用 epsilon × 平均 IDF 兜底，与 rank_bm25 一致，用于对照验证。
    """

    def __init__(
        self,
        docs: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
        idf_variant: str = "lucene",
        epsilon: float = 0.25,
    ) -> None:
        self.k1, self.b = k1, b
        self.n_docs = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / self.n_docs
        self.tf = [Counter(d) for d in docs]

        # 倒排表：词 → 含这个词的文档下标。打分时只遍历相关文档。
        self.postings: dict[str, list[int]] = {}
        for i, counts in enumerate(self.tf):
            for term in counts:
                self.postings.setdefault(term, []).append(i)

        n = self.n_docs
        self.idf: dict[str, float] = {}
        for term, docs_with_term in self.postings.items():
            df = len(docs_with_term)
            if idf_variant == "lucene":
                self.idf[term] = math.log(1 + (n - df + 0.5) / (df + 0.5))
            elif idf_variant == "okapi":
                self.idf[term] = math.log((n - df + 0.5) / (df + 0.5))
            else:
                raise ValueError(f"unknown idf_variant: {idf_variant}")
        if idf_variant == "okapi":
            floor = epsilon * sum(self.idf.values()) / len(self.idf)
            self.idf = {t: (v if v >= 0 else floor) for t, v in self.idf.items()}

    def _term_score(self, term: str, i: int) -> float:
        tf = self.tf[i][term]
        norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
        return self.idf[term] * tf * (self.k1 + 1) / (tf + norm)

    def scores(self, query: list[str]) -> list[float]:
        """对所有文档打分。问题里重复出现的词会重复计分（与 rank_bm25 一致）。"""
        out = [0.0] * self.n_docs
        for term in query:
            for i in self.postings.get(term, ()):
                out[i] += self._term_score(term, i)
        return out

    def rank(self, query: list[str]) -> list[int]:
        """返回全部文档下标，按分数从高到低；同分按下标，保证结果可复现。"""
        s = self.scores(query)
        return sorted(range(self.n_docs), key=lambda i: (-s[i], i))

    def explain(self, query: list[str], i: int) -> dict[str, float]:
        """某个文档的得分由哪些词贡献，用于失败分析。"""
        contrib: dict[str, float] = {}
        for term in query:
            if term in self.idf and self.tf[i][term]:
                contrib[term] = contrib.get(term, 0.0) + self._term_score(term, i)
        return dict(sorted(contrib.items(), key=lambda kv: -kv[1]))
