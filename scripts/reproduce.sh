#!/usr/bin/env sh
# 一键复现，分四级。越往后需要的东西越多，每一级都不依赖后面的级别。
#
#   sh scripts/reproduce.sh offline   不要数据、不要 key：跑单元测试，用仓库里的运行结果重算运行指标
#   sh scripts/reproduce.sh data      下载并校验公开 PDF → 切分（校验 chunks 哈希）→ BM25 检索评测
#   sh scripts/reproduce.sh dense     计算 bge-m3 向量（需要 embed 镜像，首次下载约 2.3GB 模型）→ 向量检索评测
#   sh scripts/reproduce.sh llm       需要 .env 里的 key 和 embed-server：在 blind_v2 和 v2.1 上跑单次 RAG、Agent、评委
#
# 评测脚本把结果写回 reports/ 下原来的文件名；在宿主机上用 git diff reports/ 看和提交的结果有没有差别。
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-python}
CHUNKS_SHA=df7f823593b313e1cf236c8c18a7f4377fd62f123a991871dea1a157dfbe8bb7
BM25="--retriever bm25 --product-filter --tokenizer search --terms insurance_terms_v2.txt"
DENSE="--retriever dense --product-filter --tokenizer search --terms insurance_terms_v2.txt"

stage=${1:-offline}
case "$stage" in
offline)
  $PY -m pytest -q
  $PY src/ops_report.py --name ops_v1 >/dev/null
  echo "运行指标已重算：reports/ops_v1.md（只读 reports/runs，不调用模型）"
  ;;
data)
  $PY scripts/fetch_sources.py
  $PY src/ingest.py
  got=$($PY -c "import hashlib;print(hashlib.sha256(open('data/processed/chunks.jsonl','rb').read()).hexdigest())")
  [ "$got" = "$CHUNKS_SHA" ] && echo "chunks 与报告使用的版本一致" || { echo "chunks 哈希不一致：$got"; exit 1; }
  $PY src/evaluate.py --run-name bm25_v4_search_dict_v2 --golden golden_v1.jsonl $BM25
  $PY -m pytest -q
  ;;
dense)
  # 在 embed 容器里运行（需要 PyTorch）。CPU 与 GPU 的浮点结果有微小差异，排名可能个别不同。
  $PY src/embed.py --golden golden_v2.jsonl --name bge-m3_index_text
  $PY src/embed.py --golden golden_blind_v1.jsonl --name bge-m3_index_text_blind
  $PY src/embed.py --golden golden_blind_v2.jsonl --name bge-m3_index_text_blind2
  ;;
dense-eval)
  $PY src/evaluate.py --run-name dense_v2_golden_v2 --golden golden_v2.jsonl $DENSE
  $PY src/evaluate.py --run-name blind_v1_dense --golden golden_blind_v1.jsonl --emb bge-m3_index_text_blind $DENSE
  $PY src/evaluate.py --run-name blind_v2_dense --golden golden_blind_v2.jsonl --emb bge-m3_index_text_blind2 $DENSE
  $PY src/evaluate.py --run-name blind_v2_1_dense --golden golden_blind_v2.1.jsonl --emb bge-m3_index_text_blind2 $DENSE
  ;;
llm)
  # 约 16×1 + 16×4 次生成调用、32 次评委调用。结果有缓存（data/processed/llm_cache），重跑不再计费。
  $PY src/answer.py --run-name gen_v3_p2_blind2 --golden golden_blind_v2.jsonl --emb bge-m3_index_text_blind2
  $PY src/agent_graph.py --run-name agent_v3_guard_blind2 --golden golden_blind_v2.jsonl --agent-prompt a2 --guard 1
  $PY src/judge.py --run gen_v3_p2_blind2 --golden golden_blind_v2.jsonl
  $PY src/judge.py --run agent_v3_guard_blind2 --golden golden_blind_v2.jsonl
  # v2.1 只改了 c06 的标注：生成全部命中上面刚写入的缓存，评委只对 c06 重新调用
  $PY src/answer.py --run-name gen_v3_p2_blind2_1 --golden golden_blind_v2.1.jsonl --emb bge-m3_index_text_blind2
  $PY src/agent_graph.py --run-name agent_v3_guard_blind2_1 --golden golden_blind_v2.1.jsonl --agent-prompt a2 --guard 1
  $PY src/judge.py --run gen_v3_p2_blind2_1 --golden golden_blind_v2.1.jsonl
  $PY src/judge.py --run agent_v3_guard_blind2_1 --golden golden_blind_v2.1.jsonl
  $PY src/ops_report.py --name ops_v1 >/dev/null
  ;;
*)
  echo "用法：sh scripts/reproduce.sh [offline|data|dense|dense-eval|llm]"; exit 2 ;;
esac
