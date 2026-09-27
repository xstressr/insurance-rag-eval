# 拆子问题 · decompose_dev

> 自动生成（2026-09-27），`src/decompose_report.py`。评委 j2；成本按列表价，含拆分调用；延迟是每题各次调用耗时之和。

| 集合 | 系统 | 可回答题答对 | 部分正确 | 拒答题 | 忠实 | 美元 / 千题 | 中位延迟（秒） |
|---|---|---|---|---|---|---|---|
| golden_v2 | 单次 RAG | 28/30 | 1 | 3/3 | 33/33 | 0.67 | 4.0 |
| golden_v2 | 拆分 d2 | 29/30 | 1 | 3/3 | 33/33 | 0.95 | 7.9 |
| golden_v2 | Agent | 29/30 | 1 | 3/3 | 32/33 | 1.79 | 7.9 |
| blind_v1 | 单次 RAG | 7/14 | 5 | 4/4 | 18/18 | 0.89 | 6.7 |
| blind_v1 | 拆分 d2 | 9/14 | 5 | 4/4 | 18/18 | 1.24 | 10.2 |
| blind_v1 | Agent | 12/14 | 2 | 4/4 | 18/18 | 3.08 | 9.0 |
| blind_v2.1 | 单次 RAG | 10/12 | 1 | 3/4 | 16/16 | 1.06 | 6.4 |
| blind_v2.1 | 拆分 d2 | 9/12 | 2 | 3/4 | 15/16 | 1.34 | 9.9 |
| blind_v2.1 | Agent | 10/12 | 2 | 4/4 | 16/16 | 3.28 | 13.6 |
| blind_v3 | 单次 RAG | 9/18 | 8 | 2/2 | 20/20 | 1.40 | 12.1 |
| blind_v3 | 拆分 d2 | 13/18 | 5 | 2/2 | 20/20 | 1.74 | 14.0 |
| blind_v3 | Agent | 15/18 | 3 | 2/2 | 20/20 | 3.62 | 17.0 |
| blind_v5 | 单次 RAG | 8/17 | 8 | 3/3 | 20/20 | 0.95 | 7.0 |
| blind_v5 | 拆分 d2 | 11/17 | 6 | 3/3 | 20/20 | 1.24 | 12.2 |
| blind_v5 | Agent | 14/17 | 3 | 3/3 | 20/20 | 2.40 | 13.3 |

## 逐题（评委判定）

| 集合 | 题 | 单次 RAG | 拆分 | Agent | 拆分的检索词 |
|---|---|---|---|---|---|
| golden_v2 | q001 | correct | correct | correct | 太保阿基米德 犹豫期 解除合同 退还保险费 |
| golden_v2 | q002 | correct | correct | correct | 泰康惠嘉保2026 等待期 疾病；泰康惠嘉保2026 保险责任 重大疾病；泰康惠嘉保2026 疾病定义 重度 |
| golden_v2 | q003 | correct | correct | correct | 国寿康宁尊享 等待期；国寿康宁尊享 等待期内 重疾 处理 |
| golden_v2 | q004 | correct | correct | correct | 太保阿基米德 保单贷款 现金价值 比例 期限 |
| golden_v2 | q005 | correct | correct | correct | 泰康惠嘉保2026 保险费的交纳与宽限期 |
| golden_v2 | q006 | correct | correct | correct | 太保阿基米德 中度疾病 保险金 给付比例；太保阿基米德 中度疾病 给付次数 |
| golden_v2 | q007 | correct | correct | correct | 太保阿基米德 轻度疾病 保险金 给付比例 次数；太保阿基米德 原位癌 疾病定义 轻度；太保阿基米德 保险金申请与给付 |
| golden_v2 | q008 | correct | correct | correct | 深度昏迷 疾病定义 呼吸机 持续时间 |
| golden_v2 | q009 | correct | correct | correct | 太保阿基米德 投保年龄 |
| golden_v2 | q010 | correct | correct | correct | 国寿康宁尊享 投保年龄 |
| golden_v2 | q011 | correct | correct | correct | 太保阿基米德 保险期间 保障期限 |
| golden_v2 | q012 | correct | correct | correct | 太保阿基米德 重度疾病 疾病定义 病种数量 |
| golden_v2 | q013 | correct | correct | correct | 国寿康宁尊享 轻度疾病 疾病定义 |
| golden_v2 | q014 | correct | correct | correct | 泰康惠嘉保2026 重大疾病 保险金 给付 比例；泰康惠嘉保2026 重大疾病 保险责任；泰康惠嘉保2026 合同效力 终止 重疾赔付后 |
| golden_v2 | q015 | correct | correct | correct | 太保阿基米德 保险责任 重大疾病 保险金给付标准；太保阿基米德 疾病定义 重度疾病；太保阿基米德 保险金申请与给付 |
| golden_v2 | q016 | correct | correct | correct | 泰康惠嘉保2026 保险金申请 医院 确诊 |
| golden_v2 | q017 | correct | correct | correct | 太保阿基米德 争议处理 诉讼时效 |
| golden_v2 | q018 | correct | correct | correct | 泰康惠嘉保2026 责任免除 自杀；泰康惠嘉保2026 保险责任 身故保险金；泰康惠嘉保2026 保险金的申请与给付 |
| golden_v2 | q019 | correct | correct | correct | 国寿康宁尊享 责任免除 酒后驾驶；国寿康宁尊享 保险责任 重大疾病；国寿康宁尊享 疾病定义 重度；国寿康宁尊享 保险金申请与给付 |
| golden_v2 | q020 | correct | correct | correct | 太保阿基米德 责任免除 遗传病；太保阿基米德 疾病定义 重度；太保阿基米德 保险金申请与给付 |
| golden_v2 | q021 | correct | correct | correct | 国寿康宁尊享 责任免除 遗传病；国寿康宁尊享 保险责任 遗传病 例外 |
| golden_v2 | q022 | correct | correct | correct | 泰康惠嘉保2026 合同效力中止与恢复 宽限期 |
| golden_v2 | q023 | correct | correct | correct | 国寿康宁尊享 保单贷款 申请条件；国寿康宁尊享 重大疾病 保险金给付 合同效力；国寿康宁尊享 保险责任 重大疾病 |
| golden_v2 | q024 | correct | correct | correct | 甲状腺癌 疾病定义 重度恶性肿瘤；甲状腺癌 疾病定义 轻度恶性肿瘤 |
| golden_v2 | q025 | correct | correct | correct | 国寿康宁尊享 年龄误告 解除合同 期限 |
| golden_v2 | q026 | correct | correct | correct | 太保阿基米德 犹豫期 起算时间；泰康惠嘉保2026 犹豫期 起算时间 |
| golden_v2 | q027 | partial | correct | correct | 太保阿基米德 等待期；泰康惠嘉保2026 等待期；国寿康宁尊享 等待期 |
| golden_v2 | q028 | correct | correct | correct | 太保阿基米德 保单贷款 额度 单次期限；国寿康宁尊享 保单贷款 额度 单次期限 |
| golden_v2 | q029 | incorrect | correct | correct | 泰康惠嘉保2026 保险费的交纳与宽限期；国寿康宁尊享 保险费的交纳与宽限期 |
| golden_v2 | q030 | correct | partial | partial | 国寿康宁尊享 犹豫期 |
| golden_v2 | q031 | correct | correct | correct | 泰康惠嘉保2026 保险费 交纳 保费 |
| golden_v2 | q032 | correct | correct | correct | 保险费 保费 产品A；保险费 保费 产品B；保险费 保费 产品C |
| golden_v2 | q033 | correct | correct | correct | 平安盛世福 等待期 |
| blind_v1 | b01 | incorrect | partial | partial | 太保阿基米德 甲状腺癌 疾病定义 重度；太保阿基米德 甲状腺癌 疾病定义 轻度；太保阿基米德 保险责任 重大疾病 轻症疾病 |
| blind_v1 | b02 | correct | correct | correct | 泰康惠嘉保 责任免除 酒后驾驶；泰康惠嘉保 疾病定义 重度；泰康惠嘉保 保险金的申请与给付 |
| blind_v1 | b03 | incorrect | partial | correct | 国寿康宁尊享 轻度疾病 保险费豁免；国寿康宁尊享 保险费的交纳 |
| blind_v1 | b04 | partial | correct | correct | 阿基米德 如实告知 未告知 高血压 理赔影响；阿基米德 责任免除 未如实告知；阿基米德 保险金申请与给付 理赔 |
| blind_v1 | b05 | partial | partial | correct | 泰康惠嘉保 保险期间 保障期限 |
| blind_v1 | b06 | correct | correct | correct | 国寿康宁尊享 保险费的交纳 宽限期 |
| blind_v1 | b07 | partial | partial | correct | 太保阿基米德 医院 确诊 指定医疗机构 |
| blind_v1 | b08 | correct | correct | correct | 疾病定义 重度 急性心肌梗死 |
| blind_v1 | b09 | correct | correct | correct | 脑中风后遗症 疾病定义 确诊180天；脑中风后遗症 轻度 中度 疾病定义；保险金申请 所需证明和资料 |
| blind_v1 | b10 | correct | correct | correct | 泰康惠嘉保 身故保险金 给付；泰康惠嘉保 保险责任；泰康惠嘉保 责任免除 |
| blind_v1 | b11 | correct | correct | correct | 国寿康宁尊享 合同内容变更 补发保险合同 |
| blind_v1 | b12 | partial | partial | partial | 阿基米德 轻度疾病 保险金 给付比例；国寿康宁尊享 轻度疾病 保险金 给付比例 |
| blind_v1 | b13 | correct | correct | correct | 泰康惠嘉保 保险事故通知 通知期限；太保阿基米德 保险事故通知 通知期限 |
| blind_v1 | b14 | correct | correct | correct | 投保年龄 年龄误告 处理 |
| blind_v1 | b15 | correct | correct | correct | 太保阿基米德 保险责任 门诊；太保阿基米德 责任免除 门诊；太保阿基米德 保险金申请与给付 |
| blind_v1 | b16 | correct | correct | correct | 泰康惠嘉保 税优健康保险 个税扣除 |
| blind_v1 | b17 | correct | correct | correct | 康宁尊享 合同效力 保险公司破产 保单处理；康宁尊享 保险金给付 未还款项；康宁尊享 解除合同 现金价值 |
| blind_v1 | b18 | partial | correct | correct | 太保阿基米德 解除合同 现金价值 |
| blind_v2.1 | c01 | correct | correct | correct | 太保阿基米德 重疾 赔付后 合同效力；太保阿基米德 重疾 多次赔付 间隔期；太保阿基米德 重疾 疾病定义 分组 |
| blind_v2.1 | c02 | correct | correct | correct | 国寿康宁尊享 重大疾病 保险责任 给付比例；国寿康宁尊享 疾病定义 重度；国寿康宁尊享 保险金申请与给付 |
| blind_v2.1 | c03 | correct | correct | correct | 泰康惠嘉保 等待期 轻症 保险责任；泰康惠嘉保 等待期 疾病定义 轻度；泰康惠嘉保 等待期 责任免除 |
| blind_v2.1 | c04 | correct | correct | correct | 严重阿尔茨海默病 疾病定义 重度；严重阿尔茨海默病 轻度 中度 疾病定义 |
| blind_v2.1 | c05 | correct | correct | correct | 重大器官移植术 器官 定义 |
| blind_v2.1 | c06 | incorrect | partial | correct | 太保阿基米德 保险责任 重大疾病 国外医院 确诊；太保阿基米德 责任免除 境外 医院；太保阿基米德 疾病定义 重度疾病；太保阿基米德 保险金申请 所需证明和资料 境外 |
| blind_v2.1 | c07 | correct | correct | correct | 太保阿基米德 投保年龄 最高；泰康惠嘉保 投保年龄 最高；国寿康宁尊享 投保年龄 最高 |
| blind_v2.1 | c08 | correct | incorrect | partial | 国寿康宁尊享 保险责任 重大疾病 豁免保险费；国寿康宁尊享 保险费的交纳 宽限期 |
| blind_v2.1 | c09 | partial | partial | partial | 泰康惠嘉保 保险金的申请与给付 理赔结论 期限 |
| blind_v2.1 | c10 | correct | correct | correct | 太保阿基米德 理赔率 |
| blind_v2.1 | c11 | correct | correct | correct | 太保阿基米德 责任免除 故意犯罪；太保阿基米德 保险责任 重大疾病；太保阿基米德 保险金的申请与给付 |
| blind_v2.1 | c12 | correct | correct | correct | 国寿康宁尊享 合同内容变更 基本保险金额 减少；国寿康宁尊享 解除合同 现金价值 |
| blind_v2.1 | c13 | correct | correct | correct | 泰康惠嘉保 保单贷款；太保阿基米德 保单贷款 |
| blind_v2.1 | c14 | partial | partial | correct | 国寿康宁尊享 投保年龄 健康告知；国寿康宁尊享 责任免除 乙型肝炎；国寿康宁尊享 疾病定义 重度 中度 轻度 肝脏疾病 |
| blind_v2.1 | c15 | correct | correct | correct | 泰康惠嘉保 保险金给付 给付方式；泰康惠嘉保 保险金申请与给付 医院账户；泰康惠嘉保 受益人 保险金给付 |
| blind_v2.1 | c16 | correct | correct | correct | 太保阿基米德 保险费的交纳 投保人 豁免；太保阿基米德 投保人 变更；太保阿基米德 合同效力中止与恢复；太保阿基米德 解除合同与现金价值 |
| blind_v3 | d01 | correct | correct | correct | 惠嘉保2026 受益人 变更 手续；惠嘉保2026 合同内容变更 申请方式；惠嘉保2026 受益人 指定 变更 申请 |
| blind_v3 | d02 | partial | correct | correct | 阿基米德2025 解除合同 现金价值；阿基米德2025 犹豫期 退还保险费；阿基米德2025 未还款项 扣除 |
| blind_v3 | d03 | correct | correct | correct | 康宁尊享2024 投保年龄 健康告知 乙肝小三阳；康宁尊享2024 保险责任 责任免除 乙肝；康宁尊享2024 保险费的交纳 加费 |
| blind_v3 | d04 | partial | correct | partial | 甲状腺癌 疾病定义 重度；甲状腺癌 疾病定义 轻度；甲状腺癌 疾病定义 中度 |
| blind_v3 | d05 | partial | partial | correct | 阿基米德2025 等待期；康宁尊享2024 等待期；等待期 疾病定义 保险责任；等待期 责任免除 合同效力 |
| blind_v3 | d06 | partial | partial | partial | 产品A 疾病定义 病种数量；产品B 疾病定义 病种数量；产品C 疾病定义 病种数量；疾病定义 重度 中度 轻度 病种数量 比较 |
| blind_v3 | d07 | correct | correct | correct | 国寿康宁尊享2024 保险责任 身故 赔付；国寿康宁尊享2024 受益人 未指定 保险金给付；国寿康宁尊享2024 责任免除 身故；国寿康宁尊享2024 保险金申请所需证明和资料 |
| blind_v3 | d08 | correct | partial | partial | 太保阿基米德2025 保险费的交纳 宽限期；太保阿基米德2025 合同效力中止与恢复 |
| blind_v3 | d09 | partial | partial | correct | 恶性肿瘤——重度 疾病定义 早期癌症 除外 |
| blind_v3 | d10 | correct | correct | correct | 泰康惠嘉保2026 保险金申请所需证明和资料；泰康惠嘉保2026 保险金的申请与给付；泰康惠嘉保2026 重度疾病定义 |
| blind_v3 | d11 | partial | correct | correct | 泰康惠嘉保2026 轻度疾病 保险责任 给付比例；国寿康宁尊享2024版 轻度疾病 保险责任 给付比例 |
| blind_v3 | d12 | correct | correct | correct | 太保阿基米德2025 受益人 变更 被保险人同意；太保阿基米德2025 受益人 变更 申请 手续；太保阿基米德2025 合同内容变更 |
| blind_v3 | d13 | correct | correct | correct | 较重急性心肌梗死 疾病定义 同时满足 |
| blind_v3 | d14 | incorrect | correct | correct | 太保阿基米德2025 心脏搭桥 开胸 疾病定义；太保阿基米德2025 轻度疾病 心脏搭桥 定义；太保阿基米德2025 保险责任 给付 |
| blind_v3 | d15 | partial | partial | correct | 太保阿基米德 受益人 变更 |
| blind_v3 | d16 | correct | correct | correct | 泰康惠嘉保2026 解除合同 现金价值；泰康惠嘉保2026 犹豫期；泰康惠嘉保2026 保险费的交纳与宽限期 |
| blind_v3 | d17 | correct | correct | correct | 严重脑中风后遗症 疾病定义；轻度脑中风后遗症 疾病定义 |
| blind_v3 | d18 | partial | correct | correct | 国寿康宁尊享 脑梗死 疾病定义 重度；国寿康宁尊享 脑梗死 轻度 中度 疾病定义；国寿康宁尊享 保险金申请 所需证明和资料；国寿康宁尊享 保险事故通知 报案 |
| blind_v3 | d19 | correct | correct | correct | 太保阿基米德 合同内容变更 联系方式 银行账户 变更 |
| blind_v3 | d20 | correct | correct | correct | 国寿康宁尊享 投保年龄 出生满28日；国寿康宁尊享 投保人 被保险人 监护人 签字 |
| blind_v5 | e01 | correct | correct | correct | 国寿康宁尊享 受益人 变更；国寿康宁尊享 受益人 变更 被保险人同意 |
| blind_v5 | e02 | correct | correct | correct | 国寿康宁尊享 保险金申请与给付 给付时限；国寿康宁尊享 保险金申请所需证明和资料；国寿康宁尊享 重度疾病 恶性肿瘤 定义；国寿康宁尊享 保险事故通知 |
| blind_v5 | e03 | partial | partial | correct | — |
| blind_v5 | e04 | correct | correct | correct | 泰康 保险费的交纳 宽限期；泰康 合同效力中止与恢复 |
| blind_v5 | e05 | partial | partial | partial | 保险金申请所需证明和资料；疾病定义 重度 恶性肿瘤；疾病定义 轻度 恶性肿瘤；保险事故通知 |
| blind_v5 | e06 | correct | correct | correct | 健康告知 住院记录 询问年限 |
| blind_v5 | e07 | partial | correct | correct | 太保阿基米德重疾险 投保年龄；太保阿基米德重疾险 保险期间 保障期限 |
| blind_v5 | e08 | partial | partial | partial | 太保阿基米德 轻症 疾病定义 赔付比例；泰康惠嘉保2026 轻症 疾病定义 赔付比例 |
| blind_v5 | e09 | correct | correct | correct | 太保阿基米德2025 解除合同 退保 线上操作；太保阿基米德2025 解除合同 现金价值 退还；太保阿基米德2025 保险金给付 退保金 到账时间 |
| blind_v5 | e10 | correct | correct | correct | 太保阿基米德2025重疾险 保险期间 保障期限 |
| blind_v5 | e11 | correct | correct | correct | 泰康惠嘉保2026 健康告知 乙肝小三阳；泰康惠嘉保2026 如实告知 义务；泰康惠嘉保2026 责任免除 未如实告知；泰康惠嘉保2026 合同解除 隐瞒病史 |
| blind_v5 | e12 | correct | correct | correct | 国寿康宁尊享2024 健康告知 告知事项；国寿康宁尊享2024 投保条件 既往病史 |
| blind_v5 | e13 | correct | correct | correct | 国寿康宁尊享2024版 合同内容变更 地址变更 微信小程序 |
| blind_v5 | e14 | correct | partial | correct | 泰康惠嘉保2026 保单贷款 利息；泰康惠嘉保2026 保单贷款 未还款项 扣除 |
| blind_v5 | e15 | partial | partial | correct | 阿基米德 重大疾病 保险金申请 给付 条件；阿基米德 疾病定义 手术 确诊 |
| blind_v5 | e16 | partial | correct | correct | 康宁尊享2024 保险金申请 证明和资料 代理人 提交；康宁尊享2024 受益人 保险金给付 对象；康宁尊享2024 保险事故通知 申请时效 |
| blind_v5 | e17 | partial | partial | partial | 太保阿基米德2025 等待期；太保阿基米德2025 等待期内确诊 保险责任；太保阿基米德2025 等待期内确诊 责任免除 |
| blind_v5 | e18 | correct | correct | correct | 严重脑中风后遗症 疾病定义 神经系统功能障碍 永久不可逆；严重脑中风后遗症 确诊180天后 理赔条件 |
| blind_v5 | e19 | partial | correct | correct | 保险金申请与给付 核定时限 赔付期限；争议处理 投诉 诉讼 |
| blind_v5 | e20 | incorrect | correct | correct | 太保阿基米德 保单贷款；太保阿基米德 保险金给付 扣除 未还贷款本息 |
