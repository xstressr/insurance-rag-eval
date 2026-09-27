"""解析保险条款 PDF，按条款结构切分，输出 data/processed/chunks.jsonl。

流程：逐页取文本 → 清洗（私用区字符、页眉页脚、页码）→ 按条款标题切成单元
→ 过长单元按列表项 / 行边界再拆 → 写出带元数据的 chunk。

用法：uv run python src/ingest.py
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "dataset" / "sources.csv"
OUT = ROOT / "data" / "processed" / "chunks.jsonl"

# 正文起始页：之前是阅读指引和目录。人工翻阅 PDF 后确定，换文档时需要重新核对。
BODY_START_PAGE = {
    "cpic_archimedes_2025": 3,
    "taikang_huijiabao_2026": 3,
    "chinalife_kangning_zunxiang_2024": 5,
    "iac_ci_definitions_2020": 1,
}

# 检索时拼在正文前面的简称，让“问的是哪个产品”能被命中。
SHORT_NAME = {
    "cpic_archimedes_2025": "太保阿基米德（2025）",
    "taikang_huijiabao_2026": "泰康惠嘉保2026",
    "chinalife_kangning_zunxiang_2024": "国寿康宁尊享（2024版）",
    "iac_ci_definitions_2020": "重疾定义规范（2020）",
}

TARGET_CHARS = 450  # 超过这个长度后，遇到列表项就断开
MAX_CHARS = 700  # 超过这个长度后，在任意行边界断开
MIN_CHARS = 20  # 短于这个长度的单元视为噪声（只剩标题等）
HEADER_RATIO = 0.3  # 在 30% 以上页面重复出现的短行视为页眉页脚

PUA = re.compile(r"[-]")  # PDF 里的私用区符号（项目符号、图标字体）
PAGE_NUMBER = re.compile(r"^[-—\s]*\d{1,3}[-—\s]*$")
NOISE_LINE = re.compile(r"^（此页正文完）$")
PUNCT = re.compile(r"[，。；：,;]")  # 标题里允许顿号，如“成立、生效”

CN_NUM = "一二三四五六七八九十百零"
# 第X条 标题（国寿）
RE_ARTICLE = re.compile(rf"^(第[{CN_NUM}]+条)(?:\s+(\S.{{0,25}}))?$")
# 1.1 / 1.4. / 3.1.1.2 单独成行，标题在下一行（太保、泰康）
RE_NUM_ONLY = re.compile(r"^([1-9]\d{0,2}(?:\.\d{1,3}){1,4})[.．]?$")
# 3.1.1.2 标题 / 2.1 正文开头（行业规范）
RE_NUM_TEXT = re.compile(r"^([1-9]\d{0,2}(?:\.\d{1,3}){1,4})[.．]?\s+(\S.*)$")
# 一级章节：“1．” 单独成行，章节名在下一行
RE_SECTION = re.compile(r"^([1-9]\d?)[.．]$")
RE_APPENDIX = re.compile(rf"^(附件[{CN_NUM}]+)\s*(\S.*)?$")
RE_PART = re.compile(r"^(个人保险基本条款|.+利益条款)$")
# 列表项开头：（1） / 1． / 一、 / a.
RE_LIST_ITEM = re.compile(rf"^(（\d+）|\d+[．.]\S|[{CN_NUM}]+、|[a-z][.．])")
CJK = re.compile(r"[一-鿿]")


@dataclass
class Unit:
    """切分前的一个条款单元。"""

    doc_id: str
    section: str  # guide（阅读指引 / 目录）或 body（正文）
    part: str = ""  # 国寿分“利益条款”和“基本条款”，其余为空
    section_title: str = ""
    clause_id: str = ""
    title: str = ""
    lines: list[tuple[int, str]] = field(default_factory=list)  # (页码, 行文本)


def is_title_like(line: str, max_len: int) -> bool:
    return bool(line) and len(line) <= max_len and not PUNCT.search(line) and bool(CJK.match(line))


def join_lines(lines: list[str]) -> str:
    """中文硬换行直接拼接；两侧都是英文或数字时补一个空格。"""
    out = ""
    for line in lines:
        if out and out[-1].isascii() and out[-1].isalnum() and line[:1].isascii() and line[:1].isalnum():
            out += " "
        out += line
    return out


def read_pages(path: Path) -> list[list[str]]:
    """逐页读取并清洗，返回每页的非空行。"""
    doc = pymupdf.open(path)
    pages = []
    for page in doc:
        lines = [PUA.sub("", raw).strip() for raw in page.get_text().splitlines()]
        pages.append([ln for ln in lines if ln])
    return pages


def drop_headers(pages: list[list[str]]) -> list[list[str]]:
    """去掉页眉页脚：把数字归一化后，在大量页面重复出现的短行。"""

    def key(line: str) -> str:
        return re.sub(rf"[0-9{CN_NUM}]+", "#", line)

    # 只统计含中文的短行：纯数字行（条款编号、页码）不能当页眉，否则会误删“1.1”这类标题
    counts = Counter(
        k for page in pages for k in {key(ln) for ln in page if len(ln) <= 40 and len(CJK.findall(ln)) >= 2}
    )
    threshold = max(3, int(len(pages) * HEADER_RATIO))
    repeated = {k for k, c in counts.items() if c >= threshold}
    cleaned = []
    for page in pages:
        cleaned.append(
            [
                ln
                for ln in page
                if key(ln) not in repeated and not PAGE_NUMBER.match(ln) and not NOISE_LINE.match(ln)
            ]
        )
    return cleaned


def segment(doc_id: str, pages: list[list[str]]) -> list[Unit]:
    """把页面行序列切成条款单元。"""
    body_start = BODY_START_PAGE[doc_id]
    units: list[Unit] = []

    # 阅读指引 / 目录：每页一个单元，打上 guide 标签
    for pno in range(1, body_start):
        units.append(Unit(doc_id, "guide", clause_id=f"guide-p{pno}", lines=[(pno, ln) for ln in pages[pno - 1]]))

    flat = [(pno, ln) for pno in range(body_start, len(pages) + 1) for ln in pages[pno - 1]]
    part = "利益条款" if doc_id.startswith("chinalife") else ""
    section_title = ""
    cur = Unit(doc_id, "body", part=part, clause_id="preamble")
    i = 0

    def start(clause_id: str, title: str) -> None:
        nonlocal cur
        units.append(cur)
        cur = Unit(doc_id, "body", part=part, section_title=section_title, clause_id=clause_id, title=title)

    while i < len(flat):
        pno, line = flat[i]
        nxt = flat[i + 1][1] if i + 1 < len(flat) else ""

        if doc_id.startswith("chinalife") and RE_PART.match(line) and len(line) <= 40:
            part = "基本条款" if line.startswith("个人保险基本条款") else "利益条款"
            start("part-title", line)
            i += 1
            continue

        if m := RE_ARTICLE.match(line):
            title = m.group(2) or ""
            if not title or is_title_like(title, 25):
                start(m.group(1), title)
                i += 1
                continue

        if (m := RE_SECTION.match(line)) and is_title_like(nxt, 20):
            section_title = nxt
            start(f"sec-{m.group(1)}", nxt)
            i += 2
            continue

        if (m := RE_NUM_ONLY.match(line)) and is_title_like(nxt, 8):
            # 标题可能被 PDF 两栏排版折成几行：“基本保险金 / 额”
            j, parts = i + 1, []
            while j < len(flat) and is_title_like(flat[j][1], 8) and len("".join(parts)) < 16:
                parts.append(flat[j][1])
                j += 1
            start(m.group(1), "".join(parts))
            i = j
            continue

        if m := RE_NUM_TEXT.match(line):
            rest = m.group(2)
            if CJK.match(rest):
                if is_title_like(rest, 30):
                    start(m.group(1), rest)
                else:  # 无标题条款：编号后直接是正文
                    start(m.group(1), "")
                    cur.lines.append((pno, rest))
                i += 1
                continue

        if m := RE_APPENDIX.match(line):
            section_title = line
            start(m.group(1), m.group(2) or "")
            i += 1
            continue

        cur.lines.append((pno, line))
        i += 1

    units.append(cur)
    return units


def split_unit(unit: Unit) -> list[list[tuple[int, str]]]:
    """过长单元再拆：够长后遇到列表项就断，太长则在任意行边界断。"""
    pieces, buf, size = [], [], 0
    for pno, line in unit.lines:
        at_item = bool(RE_LIST_ITEM.match(line))
        if buf and ((size >= TARGET_CHARS and at_item) or size >= MAX_CHARS):
            pieces.append(buf)
            buf, size = [], 0
        buf.append((pno, line))
        size += len(line)
    if buf:
        pieces.append(buf)
    return pieces


def build_chunks(unit: Unit) -> list[dict]:
    chunks = []
    for k, piece in enumerate(split_unit(unit)):
        text = join_lines([ln for _, ln in piece])
        if len(text) < MIN_CHARS or len(CJK.findall(text)) < 0.2 * len(text):
            continue  # 太短，或基本是数字表格
        context = "｜".join(
            x for x in [SHORT_NAME[unit.doc_id], unit.part, unit.section_title, f"{unit.clause_id} {unit.title}".strip()] if x
        )
        chunks.append(
            {
                "chunk_id": f"{unit.doc_id}::{unit.part or '-'}::{unit.clause_id}#{k}",
                "doc_id": unit.doc_id,
                "section": unit.section,
                "part": unit.part,
                "section_title": unit.section_title,
                "clause_id": unit.clause_id,
                "title": unit.title,
                "sub_index": k,
                "page_start": piece[0][0],
                "page_end": piece[-1][0],
                "n_chars": len(text),
                "text": text,
                "index_text": f"{context}\n{text}",
            }
        )
    return chunks


def main() -> None:
    with SOURCES.open(encoding="utf-8") as f:
        sources = list(csv.DictReader(f))

    all_chunks = []
    for src in sources:
        doc_id = src["doc_id"]
        pages = drop_headers(read_pages(ROOT / src["file"]))
        units = segment(doc_id, pages)
        chunks = [c for u in units for c in build_chunks(u)]
        all_chunks.extend(chunks)

        body = [c for c in chunks if c["section"] == "body"]
        lengths = sorted(c["n_chars"] for c in chunks)
        clause_ids = {(c["part"], c["clause_id"]) for c in body}
        print(
            f"{doc_id}: pages={len(pages)} units={len(units)} chunks={len(chunks)} "
            f"(guide={len(chunks) - len(body)}, body={len(body)}) clauses={len(clause_ids)} "
            f"chars p10/p50/p90/max={lengths[len(lengths) // 10]}/{lengths[len(lengths) // 2]}/"
            f"{lengths[len(lengths) * 9 // 10]}/{lengths[-1]}"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # 固定 LF：Windows 和 Linux 的输出逐字节相同，复现脚本可以直接比对哈希
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(all_chunks)} chunks -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
