"""用 bge-reranker-v2-m3 给每个“问题 × 文本块”打相关性分数，存到 data/processed/rerank/<name>/。

需要 PyTorch，所以和 embed.py 一样在 Miniconda 环境里运行：
    C:\\Users\\xstre\\miniconda3\\python.exe src/rerank.py --golden golden_v2.jsonl --name golden_v2
    C:\\Users\\xstre\\miniconda3\\python.exe src/rerank.py --golden golden_blind_v1.jsonl --name blind_v1

重排模型（cross-encoder）和向量模型（bi-encoder）的区别：
- 向量模型把问题和文本块分别编码成向量，再比余弦。快，可以提前算好所有块的向量；
- 重排模型把“问题 + 文本块”拼在一起输入，逐字交叉注意力后输出一个相关性分数。准，但每一对都要单独算。
所以通常用向量检索粗选，再用重排精排。这里语料小（565 块），直接给全部组合打分，
评测时再决定取向量检索的前多少名来重排。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
DATASET = ROOT / "dataset"
OUT_DIR = ROOT / "data" / "processed" / "rerank"
DEFAULT_MODEL = os.environ.get("BGE_RERANKER_DIR", r"D:\Projects\models\bge-reranker-v2-m3")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, help="dataset 下的评测集文件名")
    ap.add_argument("--name", required=True, help="输出目录名")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--field", default="index_text", choices=["index_text", "text"])
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="只算前 N 题，用于测速")
    args = ap.parse_args()

    chunks = load_jsonl(CHUNKS)
    golden = load_jsonl(DATASET / args.golden)[: args.limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(
        args.model, device=device, max_length=args.max_len, model_kwargs={"torch_dtype": torch.float16}
    )

    t0 = time.time()
    docs = [c[args.field] for c in chunks]
    scores = np.zeros((len(golden), len(chunks)), dtype=np.float32)
    for qi, q in enumerate(golden):
        pairs = [(q["question"], d) for d in docs]
        # 不做 sigmoid：只用来排序，原始 logit 足够，也保留分数差距
        scores[qi] = model.predict(pairs, batch_size=args.batch, show_progress_bar=False, activation_fn=torch.nn.Identity())
        print(f"{q['id']} done ({time.time() - t0:.0f}s)", flush=True)
    seconds = round(time.time() - t0, 1)

    out = OUT_DIR / args.name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "scores.npy", scores)
    meta = {
        "model": Path(args.model).name,
        "field": args.field,
        "max_len": args.max_len,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "seconds": seconds,
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "question_ids": [q["id"] for q in golden],
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(golden)} questions x {len(chunks)} chunks, {seconds}s on {meta['device']} -> {out}")


if __name__ == "__main__":
    main()
