"""本地向量服务：bge-m3 常驻 GPU，给 uv 环境里的 Agent 实时编码查询。

需要 PyTorch，所以在 Miniconda 里运行：
    C:\\Users\\xstre\\miniconda3\\python.exe src/embed_server.py

只监听 127.0.0.1。接口：
    POST /embed  {"texts": ["..."]}  →  {"vectors": [[...], ...]}   （L2 归一化，点积即余弦）
    GET  /health                      →  {"model": "bge-m3", "dim": 1024}
只用标准库 http.server，不引入 Web 框架。
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from sentence_transformers import SentenceTransformer

HOST, PORT = "127.0.0.1", int(os.environ.get("EMBED_PORT", "8765"))
MODEL_DIR = os.environ.get("BGE_M3_DIR", r"D:\Projects\models\bge-m3")

model = SentenceTransformer(MODEL_DIR, device="cuda" if torch.cuda.is_available() else "cpu")
model.max_seq_length = 1024


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"model": os.path.basename(MODEL_DIR), "dim": model.get_sentence_embedding_dimension()})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/embed":
            self._send(404, {"error": "not found"})
            return
        try:
            texts = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["texts"]
            assert isinstance(texts, list) and all(isinstance(t, str) for t in texts) and len(texts) <= 64
        except Exception:  # noqa: BLE001
            self._send(400, {"error": "body must be {\"texts\": [str, ...]} with at most 64 items"})
            return
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=16)
        self._send(200, {"vectors": vecs.tolist()})

    def log_message(self, fmt: str, *args) -> None:  # 默认每个请求打一行日志，太吵
        pass


if __name__ == "__main__":
    print(f"embed server on http://{HOST}:{PORT} ({MODEL_DIR})", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
