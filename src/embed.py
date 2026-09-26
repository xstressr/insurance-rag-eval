"""用 bge-m3 计算文本块和评测题的向量，存到 data/processed/emb/<name>/。

需要 PyTorch，所以在 Miniconda 环境里运行（uv 环境只负责读取结果做评测）：
    C:\\Users\\xstre\\miniconda3\\python.exe src/embed.py

向量做了 L2 归一化，所以点积就是余弦相似度。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
GOLDEN = ROOT / "dataset" / "golden_v1.jsonl"
EMB_DIR = ROOT / "data" / "processed" / "emb"
DEFAULT_MODEL = os.environ.get("BGE_M3_DIR", r"D:\Projects\models\bge-m3")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--field", default="index_text", choices=["index_text", "text"])
    ap.add_argument("--max-len", type=int, default=1024)  # 最长的块约 750 字，1024 个 token 足够
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default=None, help="输出目录名，默认 bge-m3_<field>")
    ap.add_argument("--golden", default=GOLDEN.name, help="dataset 下的评测集文件名")
    args = ap.parse_args()

    chunks = load_jsonl(CHUNKS)
    golden = load_jsonl(GOLDEN.parent / args.golden)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = args.max_len

    t0 = time.time()
    chunk_vecs = model.encode(
        [c[args.field] for c in chunks], batch_size=args.batch, normalize_embeddings=True, show_progress_bar=True
    )
    # bge-m3 的查询不需要加指令前缀（bge v1.5 系列才需要）
    question_vecs = model.encode([q["question"] for q in golden], batch_size=args.batch, normalize_embeddings=True)
    seconds = round(time.time() - t0, 1)

    out = EMB_DIR / (args.name or f"bge-m3_{args.field}")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "chunks.npy", chunk_vecs.astype(np.float32))
    np.save(out / "questions.npy", question_vecs.astype(np.float32))
    meta = {
        "model": Path(args.model).name,
        "field": args.field,
        "max_len": args.max_len,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "dim": int(chunk_vecs.shape[1]),
        "seconds": seconds,
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "question_ids": [q["id"] for q in golden],
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(chunks)} chunks + {len(golden)} questions, dim={meta['dim']}, {seconds}s on {meta['device']} -> {out}")


if __name__ == "__main__":
    main()
