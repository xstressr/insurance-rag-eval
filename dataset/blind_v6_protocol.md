# golden_blind_v6 协议（先提交，后生成）

> 2026-09-27 写定。用途：独立检验拆子问题（`reports/decompose_v1_protocol.md`）。拆分的做法和提示词在这批题生成之前定下并提交。

## 与 blind_v5 协议相同的部分

出题提示词（`WRITER_PROMPT`）、每个模型 15 题、格式过滤、去重阈值 0.90、按模型分层抽样 7 / 7 / 6、标注方法、评委 j2，都与 `dataset/blind_v3_protocol.md`、`dataset/blind_v5_protocol.md` 相同。

## 不同的部分

- **出题模型又换了一批家族**：`gpt-6-luna`、`longcat-2.0`、`omen-alpha`（`src/gen_questions.py` 的 `CONFIGS["v6"]`）。
  - 仍然避开 DeepSeek（生成、拆分）和 GLM（评委），也不用 MiniMax；前两批用过的 kimi、qwen、muse-spark、mimo、hy、grok 都不再用。
  - `omen-alpha` 的底座来源无法核实。如果它是 DeepSeek 或 GLM 的衍生模型，回避的效果会打折扣，和 v3 的 muse-spark 一样。
  - **事先定好的替补**：某个模型调用失败，或者合格的题不足 10 道，就换成 `kimi-k2.6`，只换一次，写进执行记录。
  - 如果模型不支持 chat 接口，就自动改用 Responses 接口，此时 temperature 为服务端默认值。
- **去重对照集增加了 golden_blind_v5。**
- **抽样种子**：`2026092706`；编号 f01–f20。

## 用法

1. 单次 RAG（gen_v3_p2）、拆分（gen_v4_d2）、Agent（agent_v3_guard）在 v6 上各跑一次，评委打分。
2. 不再修改拆分的提示词和参数。
3. 按 `reports/decompose_v1_protocol.md` 的预测逐条核对。

## 执行记录（生成与筛选之后补记）
