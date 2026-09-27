# 评测环境：与本地 uv 环境相同的依赖（uv.lock 锁定版本），不含 PyTorch。
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.16 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project
# 代码在运行时通过 compose 挂载到 /app，结果直接写回宿主机的 reports/ 和 data/；这里的拷贝只供单独 docker run 使用
COPY . .
CMD ["sh", "scripts/reproduce.sh", "offline"]
