# 路由评估 · route_v1_on_v5

> 自动生成（2026-09-27），`src/route.py`。离线评估：按规则逐题选用单次 RAG 或 Agent 已有的答案和评委判定，不调用模型。级联类策略的成本和延迟 = 单次 RAG + 被升级题的 Agent。oracle 是事后挑最优，只作上限参考。

## v5（20 题）

| 策略 | 可回答题答对 | 部分正确 | 拒答题 | 忠实 | 走 Agent 的题 | 美元 / 千题 | 中位延迟（秒） |
|---|---|---|---|---|---|---|---|
| always_rag | 8/17 | 8 | 3/3 | 20/20 | 0/20 | 0.95 | 7.0 |
| always_agent | 14/17 | 3 | 3/3 | 20/20 | 20/20 | 2.40 | 13.3 |
| shape | 9/17 | 7 | 3/3 | 20/20 | 6/20 | 1.35 | 8.7 |
| cascade | 14/17 | 3 | 3/3 | 20/20 | 14/20 | 2.63 | 17.4 |
| shape_or_cascade | 14/17 | 3 | 3/3 | 20/20 | 16/20 | 2.62 | 16.5 |
| oracle | 14/17 | 3 | 3/3 | 20/20 | 6/20 | 1.35 | 9.3 |

逐题路由（shape / cascade 是否触发；两套系统的评委判定）：

| 题 | 集合 | shape | cascade | 单次 RAG | Agent | 问题 |
|---|---|---|---|---|---|---|
| e01 | golden_blind_v5 |  |  | correct | correct | 国寿康宁尊享的身故受益人能改成我妹妹吗？原来写的是我妈，改的时候需不需要我妈同意 |
| e02 | golden_blind_v5 |  | ✓ | correct | correct | 确诊了乳腺癌，国寿康宁尊享的重疾金多久能到账？要不要先自己垫付 |
| e03 | golden_blind_v5 |  | ✓ | partial | correct | 泰康惠嘉保2026这个，轻症和中症是怎么赔的？是按保额的百分比吗，能赔几次啊 |
| e04 | golden_blind_v5 |  | ✓ | correct | correct | 这个月工资还没发，泰康那个保费晚交几天有事吗？宽限期是多久 |
| e05 | golden_blind_v5 |  |  | partial | partial | 上个月刚查出肺上有结节，医生说可能是早期的，这种情况下申请重疾理赔需要准备什么材 |
| e06 | golden_blind_v5 |  | ✓ | correct | correct | 健康告知里问到的住院记录是查几年内的？我十年前住院的事还用说吗 |
| e07 | golden_blind_v5 | ✓ | ✓ | partial | correct | 我想给我爸买太保的阿基米德重疾险，他今年55了还能买不？能保到多大岁数？ |
| e08 | golden_blind_v5 | ✓ | ✓ | partial | partial | 太保阿基米德和泰康惠嘉保2026，这俩重疾险哪个轻症赔付比例高一点？我想比较下。 |
| e09 | golden_blind_v5 | ✓ | ✓ | correct | correct | 我买了太保阿基米德2025重疾险，现在想退保，线上能操作吗？退的钱啥时候到？ |
| e10 | golden_blind_v5 | ✓ |  | correct | correct | 太保阿基米德2025重疾险，这玩意儿保终身吗？还是只能保到70岁？ |
| e11 | golden_blind_v5 |  | ✓ | correct | correct | 泰康惠嘉保2026重疾险，我有乙肝小三阳，买之前健康告知怎么填？ |
| e12 | golden_blind_v5 |  |  | correct | correct | 买国寿康宁尊享2024前，我做过阑尾手术，健康告知要填吗？ |
| e13 | golden_blind_v5 |  | ✓ | correct | correct | 国寿康宁尊享2024版重疾险，我搬家了地址变了，在微信小程序里能改不？ |
| e14 | golden_blind_v5 | ✓ | ✓ | correct | correct | 泰康惠嘉保2026，保单借款利息多少啊？借了之后忘了还咋办？ |
| e15 | golden_blind_v5 |  | ✓ | partial | correct | 阿基米德已经买了，医院通知我下周手术，重疾是确诊就能报，还是做完手术才能报？ |
| e16 | golden_blind_v5 |  | ✓ | partial | correct | 我爸买的康宁尊享2024，他住院昏迷了，理赔材料我能代交吗，钱打给谁？ |
| e17 | golden_blind_v5 | ✓ |  | partial | partial | 太保阿基米德2025这个重疾，等待期到底几天啊？等着等着查出来病了赔不赔？ |
| e18 | golden_blind_v5 |  |  | correct | correct | 那个规范里说的严重脑中风后遗症，到底要瘫成什么样、过多久才算能赔啊？ |
| e19 | golden_blind_v5 |  | ✓ | partial | correct | 阿基米德理赔他们拖着不批，我能不能去投诉，一般多久必须给个结果？ |
| e20 | golden_blind_v5 |  | ✓ | incorrect | correct | 太保阿基米德能不能保单贷款啊，贷出来以后生病了，赔付有没有影响？ |

