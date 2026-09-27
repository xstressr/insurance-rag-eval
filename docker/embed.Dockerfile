# 向量环境：CPU 版 PyTorch + sentence-transformers，版本与本地 Miniconda 环境一致。
# 模型权重不打进镜像，首次运行时下载到 hf-cache 卷（BAAI/bge-m3，约 2.3GB）。
FROM python:3.12-slim
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.7.0 \
 && pip install --no-cache-dir sentence-transformers==6.1.0
ENV PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 HF_HOME=/hf-cache
WORKDIR /app
CMD ["python", "src/embed_server.py"]
