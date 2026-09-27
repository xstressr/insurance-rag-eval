# golden_blind_v5 协议（先提交，后生成）

> 2026-09-27 写定。用途：独立检验路由 v1（`reports/route_v1_protocol.md`）和路由 v2（`reports/route_v2_protocol.md`）。两套路由的规则都在这批题生成之前定下并提交。

编号说明：v4 留给学习者本人写的题，所以这批叫 v5。

## 与 blind_v3 协议相同的部分

出题提示词（`WRITER_PROMPT`）、每个模型 15 题、格式过滤、去重阈值 0.90、按模型分层抽样 7 / 7 / 6、标注方法、两套系统的配置（gen_v3_p2 / agent_v3_guard）、评委 j2，都与 `dataset/blind_v3_protocol.md` 相同。

## 不同的部分

- **出题模型换了一批家族**：`mimo-v2.6-pro`、`hy3`、`grok-4.7`（`src/gen_questions.py` 的 `CONFIGS["v5"]`）。
  - 仍然避开 DeepSeek（生成）和 GLM（评委），也不用 MiniMax。
  - 如果某个模型不支持 chat 接口，就自动改用 Responses 接口，此时 temperature 为服务端默认值，会在执行记录里写明。
- **去重对照集增加了 golden_blind_v3。**
- **抽样种子**：`2026092705`；编号 e01–e20。

## 用法

1. 两套系统在 v5 上各跑一次，评委打分。
2. 路由 v1 和路由 v2 按已提交的规则，逐题选用答案，不再修改任何规则或提示词。
3. 与“全部走单次 RAG”“全部走 Agent”“理论上限”一起报告。
