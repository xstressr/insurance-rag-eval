# 路由评估 · route_v1

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

## v2.1+v3（36 题）

| 策略 | 可回答题答对 | 部分正确 | 拒答题 | 忠实 | 走 Agent 的题 | 美元 / 千题 | 中位延迟（秒） |
|---|---|---|---|---|---|---|---|
| always_rag | 19/30 | 9 | 5/6 | 36/36 | 0/36 | 1.25 | 8.2 |
| always_agent | 25/30 | 5 | 6/6 | 36/36 | 36/36 | 3.47 | 15.7 |
| shape | 21/30 | 7 | 5/6 | 36/36 | 14/36 | 1.84 | 9.9 |
| cascade | 24/30 | 6 | 6/6 | 36/36 | 23/36 | 3.89 | 23.3 |
| shape_or_cascade | 23/30 | 7 | 6/6 | 36/36 | 25/36 | 3.27 | 17.0 |
| oracle | 27/30 | 3 | 6/6 | 36/36 | 9/36 | 1.98 | 12.3 |

逐题路由（shape / cascade 是否触发；两套系统的评委判定）：

| 题 | 集合 | shape | cascade | 单次 RAG | Agent | 问题 |
|---|---|---|---|---|---|---|
| c01 | golden_blind_v2.1 | ✓ |  | correct | correct | 太保阿基米德赔过一次重疾以后，合同还在吗？以后再得别的重疾还能赔吗？ |
| c02 | golden_blind_v2.1 |  |  | correct | correct | 国寿康宁尊享确诊重疾，最多能拿到保额的几倍？ |
| c03 | golden_blind_v2.1 |  | ✓ | correct | correct | 泰康惠嘉保在等待期里查出来轻症，会怎么样？ |
| c04 | golden_blind_v2.1 |  |  | correct | correct | 按行业规范，严重阿尔茨海默病要满足什么条件才算重疾？ |
| c05 | golden_blind_v2.1 |  |  | correct | correct | 按行业规范，重大器官移植术包括哪几种器官？ |
| c06 | golden_blind_v2.1 |  | ✓ | incorrect | correct | 买了太保阿基米德，出国期间在国外医院确诊重疾，能赔吗？ |
| c07 | golden_blind_v2.1 | ✓ | ✓ | correct | correct | 太保阿基米德、泰康惠嘉保、国寿康宁尊享，最晚分别多大年纪还能投保？ |
| c08 | golden_blind_v2.1 |  | ✓ | correct | partial | 国寿康宁尊享交费期间得了重疾，后面的保费还要交吗？ |
| c09 | golden_blind_v2.1 |  |  | partial | partial | 泰康惠嘉保材料交齐以后，保险公司最晚多久要给理赔结论？ |
| c10 | golden_blind_v2.1 |  | ✓ | correct | correct | 太保阿基米德去年的理赔率是多少？ |
| c11 | golden_blind_v2.1 |  |  | correct | correct | 太保阿基米德的被保险人故意犯罪导致重疾，赔不赔？ |
| c12 | golden_blind_v2.1 |  |  | correct | correct | 国寿康宁尊享买了以后想把保额降低一点，可以吗？ |
| c13 | golden_blind_v2.1 | ✓ | ✓ | correct | correct | 泰康惠嘉保和太保阿基米德，能不能用保单借钱？ |
| c14 | golden_blind_v2.1 |  | ✓ | partial | correct | 我有乙肝小三阳，能买国寿康宁尊享吗？ |
| c15 | golden_blind_v2.1 |  | ✓ | correct | correct | 泰康惠嘉保的理赔款可以直接打到医院账户吗？ |
| c16 | golden_blind_v2.1 |  | ✓ | correct | correct | 太保阿基米德投保人和被保险人不是同一个人，投保人去世了，后面的保费怎么办？ |
| d01 | golden_blind_v3 | ✓ | ✓ | correct | correct | 我买的惠嘉保2026，受益人之前写的前妻，现在想改成我儿子，手机上能改吗？要啥手 |
| d02 | golden_blind_v3 | ✓ | ✓ | partial | correct | 阿基米德2025我交了三年了，现在手头紧想退保，能退回来多少钱？是不是只能退现金 |
| d03 | golden_blind_v3 | ✓ | ✓ | correct | correct | 我有乙肝小三阳但肝功能一直正常，这种情况买康宁尊享2024能买吗？会不会要加费？ |
| d04 | golden_blind_v3 | ✓ | ✓ | partial | partial | 听说2020年重疾新规出来以后，甲状腺癌不一定按重疾全赔了？具体是怎么分的？ |
| d05 | golden_blind_v3 | ✓ | ✓ | partial | correct | 阿基米德2025和康宁尊享2024这两款重疾险，等待期分别是多久啊？等待期内查出 |
| d06 | golden_blind_v3 | ✓ | ✓ | partial | partial | 这三款重疾保的病种数量差得多吗？是不是病种越多就越值得买？ |
| d07 | golden_blind_v3 | ✓ | ✓ | correct | correct | 我爸去世了，他生前买过国寿康宁尊享2024，人走了这个保险能赔钱吗？要是没指定受 |
| d08 | golden_blind_v3 | ✓ |  | correct | partial | 太保阿基米德2025忘交保费了，宽限期多少天？过了会直接失效吗？ |
| d09 | golden_blind_v3 |  |  | partial | correct | 2020版重疾规范里的“恶性肿瘤——重度”是不是不包括早期癌症？ |
| d10 | golden_blind_v3 |  |  | correct | correct | 泰康惠嘉保2026确诊重疾后，理赔一般要交哪些材料？ |
| d11 | golden_blind_v3 | ✓ | ✓ | partial | correct | 泰康惠嘉保2026和国寿康宁尊享2024版，轻症这块哪个赔得更多？ |
| d12 | golden_blind_v3 | ✓ | ✓ | correct | correct | 太保阿基米德2025想把受益人改成孩子，需要本人同意吗？怎么改？ |
| d13 | golden_blind_v3 |  |  | correct | correct | 较重急性心肌梗死这个定义，是必须同时满足好几项指标吗？ |
| d14 | golden_blind_v3 |  | ✓ | incorrect | correct | 买了太保阿基米德2025，做了心脏搭桥手术，这种是一定要开胸才能赔吗？ |
| d15 | golden_blind_v3 |  |  | partial | correct | 我给我老公买的太保阿基米德，受益人写的是我，以后想改成我儿子可以吗，要咋操作？ |
| d16 | golden_blind_v3 |  |  | correct | correct | 泰康惠嘉保2026我买了不到一年，现在觉得压力大想退掉，能退多少钱啊，是全退吗？ |
| d17 | golden_blind_v3 |  | ✓ | correct | correct | 我看重疾里有个严重脑中风后遗症，到底要多严重才算啊，像轻微中风手脚还能动的那种算 |
| d18 | golden_blind_v3 |  | ✓ | partial | correct | 我爸买的是国寿康宁尊享，前几天脑梗住院了，这种情况能赔吗，我要先打电话报案吗？ |
| d19 | golden_blind_v3 |  | ✓ | correct | correct | 我换手机号跟银行卡了，之前买的太保阿基米德保单信息咋改啊，必须去柜台办吗，手机上 |
| d20 | golden_blind_v3 | ✓ | ✓ | correct | correct | 刚出生的小宝宝能买国寿康宁尊享吗？要他爸妈都签字吗，还是一个人签就行？ |

