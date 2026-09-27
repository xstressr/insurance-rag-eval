"""下载 dataset/sources.csv 里的公开条款 PDF 到 data/raw/，并用 dataset/sources_sha256.txt 校验版本。

    python scripts/fetch_sources.py            # 下载缺失的文件并校验
    python scripts/fetch_sources.py --check    # 只校验，不下载

PDF 不随仓库分发（公开可访问不等于允许转载），所以要自己从官网下载。
官网换了文件，校验就会失败：说明拿到的版本与报告里用的不同，结果不保证可复现。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER_AGENT = "insurance-rag-eval/0.1 (+reproduction)"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验本地文件")
    args = ap.parse_args()

    expected = dict(
        reversed(line.split()) for line in (ROOT / "dataset" / "sources_sha256.txt").read_text(encoding="utf-8").splitlines() if line.strip()
    )
    ok = True
    with (ROOT / "dataset" / "sources.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = ROOT / row["file"]
            if not path.exists() and not args.check:
                path.parent.mkdir(parents=True, exist_ok=True)
                url = urllib.parse.quote(row["url"], safe=":/?=&%")  # 国寿的链接里有中文和括号
                print(f"下载 {row['doc_id']} ← {row['url']}")
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    path.write_bytes(resp.read())
            if not path.exists():
                print(f"缺失 {row['file']}")
                ok = False
                continue
            got = sha256(path)
            match = got == expected[row["file"]]
            ok &= match
            print(f"{'一致' if match else '不一致'} {row['file']} sha256={got[:16]}…")
    if not ok:
        sys.exit("校验未通过：文件缺失或与报告使用的版本不同。")


if __name__ == "__main__":
    main()
