# 路由评估 · route_dev

> 自动生成（2026-09-27），`src/route.py`。离线评估：按规则逐题选用单次 RAG 或 Agent 已有的答案和评委判定，不调用模型。级联类策略的成本和延迟 = 单次 RAG + 被升级题的 Agent。oracle 是事后挑最优，只作上限参考。

## dev（51 题）

| 策略 | 可回答题答对 | 部分正确 | 拒答题 | 忠实 | 走 Agent 的题 | 美元 / 千题 | 中位延迟（秒） |
|---|---|---|---|---|---|---|---|
| always_rag | 35/44 | 6 | 7/7 | 51/51 | 0/51 | 0.75 | 5.0 |
| always_agent | 41/44 | 3 | 7/7 | 50/51 | 51/51 | 2.25 | 8.7 |
| shape | 37/44 | 5 | 7/7 | 51/51 | 14/51 | 1.19 | 5.5 |
| cascade | 42/44 | 2 | 7/7 | 50/51 | 20/51 | 1.81 | 10.2 |
| shape_or_cascade | 42/44 | 2 | 7/7 | 50/51 | 29/51 | 1.91 | 9.1 |
| oracle | 42/44 | 2 | 7/7 | 51/51 | 8/51 | 1.06 | 5.3 |

逐题路由（shape / cascade 是否触发；两套系统的评委判定）：

| 题 | 集合 | shape | cascade | 单次 RAG | Agent | 问题 |
|---|---|---|---|---|---|---|
| q001 | golden_v2 |  |  | correct | correct | 太保阿基米德买了之后后悔了，多少天内可以退保拿回保费？ |
| q002 | golden_v2 |  |  | correct | correct | 泰康惠嘉保2026生病（不是意外）要等多久，得重疾才能赔？ |
| q003 | golden_v2 | ✓ |  | correct | correct | 国寿康宁尊享的等待期多长？等待期内得重疾怎么处理？ |
| q004 | golden_v2 | ✓ |  | correct | correct | 太保阿基米德保单贷款最多能借现金价值的几成？一次最长借多久？ |
| q005 | golden_v2 |  |  | correct | correct | 泰康惠嘉保2026保费到期没交，有多少天宽限？ |
| q006 | golden_v2 | ✓ |  | correct | correct | 太保阿基米德的中症按保额的多少赔？最多赔几次？ |
| q007 | golden_v2 | ✓ |  | correct | correct | 太保阿基米德的轻症赔多少比例？最多几次？原位癌能赔几次？ |
| q008 | golden_v2 |  |  | correct | correct | 按行业规范，深度昏迷要持续用呼吸机多久才算重疾？ |
| q009 | golden_v2 | ✓ |  | correct | correct | 太保阿基米德最大多少岁还能投保？最小呢？ |
| q010 | golden_v2 |  |  | correct | correct | 国寿康宁尊享什么年龄的人可以买？ |
| q011 | golden_v2 |  |  | correct | correct | 太保阿基米德可以选哪几种保障期限？ |
| q012 | golden_v2 |  |  | correct | correct | 太保阿基米德一共保多少种重疾？ |
| q013 | golden_v2 |  |  | correct | correct | 国寿康宁尊享保几种轻度疾病？ |
| q014 | golden_v2 | ✓ |  | correct | correct | 泰康惠嘉保2026确诊重疾后赔多少钱？赔完合同还在吗？ |
| q015 | golden_v2 |  |  | correct | correct | 太保阿基米德得了重疾，保险金按什么标准算？ |
| q016 | golden_v2 |  |  | correct | correct | 泰康惠嘉保2026理赔时对确诊医院有什么要求？ |
| q017 | golden_v2 |  |  | correct | correct | 太保阿基米德如果要打官司索赔，时效是几年？ |
| q018 | golden_v2 |  |  | correct | correct | 买了泰康惠嘉保2026，第二年自杀了，身故保险金赔不赔？ |
| q019 | golden_v2 |  |  | correct | correct | 国寿康宁尊享：酒驾出车祸导致重疾，能赔吗？ |
| q020 | golden_v2 |  |  | correct | correct | 太保阿基米德：遗传病导致的重疾赔不赔？ |
| q021 | golden_v2 | ✓ |  | correct | correct | 国寿康宁尊享：遗传病是不是一律不赔？有没有例外？ |
| q022 | golden_v2 |  |  | correct | correct | 泰康惠嘉保2026停交保费后，多久之内还能把合同恢复？ |
| q023 | golden_v2 |  |  | correct | correct | 国寿康宁尊享确诊重疾以后，还能申请保单借款吗？ |
| q024 | golden_v2 |  |  | correct | correct | 按行业规范，一期甲状腺癌算不算重度恶性肿瘤？ |
| q025 | golden_v2 |  | ✓ | correct | correct | 国寿康宁尊享：投保时年龄报错了，保险公司最晚什么时候能解除合同？ |
| q026 | golden_v2 | ✓ |  | correct | correct | 太保阿基米德和泰康惠嘉保2026的犹豫期分别多长？起算时间一样吗？ |
| q027 | golden_v2 | ✓ | ✓ | partial | correct | 三款重疾险（太保阿基米德、泰康惠嘉保2026、国寿康宁尊享）的等待期各是多久？ |
| q028 | golden_v2 | ✓ |  | correct | correct | 太保阿基米德和国寿康宁尊享的保单贷款，额度和单次期限有什么区别？ |
| q029 | golden_v2 | ✓ | ✓ | incorrect | correct | 泰康惠嘉保2026和国寿康宁尊享，谁的宽限期更长？ |
| q030 | golden_v2 |  |  | correct | partial | 国寿康宁尊享的犹豫期是多少天？ |
| q031 | golden_v2 |  | ✓ | correct | correct | 30岁买泰康惠嘉保2026，一年保费多少钱？ |
| q032 | golden_v2 | ✓ | ✓ | correct | correct | 三款产品里哪个保费最便宜？ |
| q033 | golden_v2 |  | ✓ | correct | correct | 平安盛世福的等待期是多久？ |
| b01 | golden_blind_v1 |  | ✓ | incorrect | partial | 太保阿基米德如果查出甲状腺癌，是按重疾赔还是按轻症赔？ |
| b02 | golden_blind_v1 |  | ✓ | correct | correct | 泰康惠嘉保要是喝了酒开车出事，得了重病还能赔吗？ |
| b03 | golden_blind_v1 |  | ✓ | incorrect | correct | 国寿康宁尊享得了轻症以后，后面的保费还要继续交吗？ |
| b04 | golden_blind_v1 |  | ✓ | partial | correct | 买阿基米德的时候没告诉保险公司自己有高血压，以后理赔会怎么样？ |
| b05 | golden_blind_v1 |  | ✓ | partial | correct | 泰康惠嘉保最多能保到多少岁？ |
| b06 | golden_blind_v1 |  |  | correct | correct | 国寿康宁尊享到期没钱交保费，有多久的缓冲时间？ |
| b07 | golden_blind_v1 |  | ✓ | partial | correct | 太保阿基米德要在什么样的医院确诊才算数？ |
| b08 | golden_blind_v1 |  |  | correct | correct | 心梗要严重到什么程度才算重疾？ |
| b09 | golden_blind_v1 |  |  | correct | correct | 脑中风之后要过多久还有后遗症，才能按重疾理赔？ |
| b10 | golden_blind_v1 |  |  | correct | correct | 泰康惠嘉保如果人没了，赔多少钱？ |
| b11 | golden_blind_v1 |  | ✓ | correct | correct | 国寿康宁尊享的保单弄丢了怎么办？ |
| b12 | golden_blind_v1 | ✓ | ✓ | partial | partial | 阿基米德和国寿康宁尊享，哪个轻症赔得多？ |
| b13 | golden_blind_v1 | ✓ | ✓ | correct | correct | 泰康惠嘉保和太保阿基米德，出事以后多长时间内要通知保险公司？ |
| b14 | golden_blind_v1 |  | ✓ | correct | correct | 投保的时候年龄填错了，保险公司会怎么处理？ |
| b15 | golden_blind_v1 |  | ✓ | correct | correct | 太保阿基米德能不能报销平时门诊看病的钱？ |
| b16 | golden_blind_v1 |  | ✓ | correct | correct | 买泰康惠嘉保能不能抵个税？ |
| b17 | golden_blind_v1 |  | ✓ | correct | correct | 要是国寿倒闭了，我的康宁尊享保单怎么办？ |
| b18 | golden_blind_v1 |  | ✓ | partial | correct | 太保阿基米德中途退保能拿回多少钱？ |

