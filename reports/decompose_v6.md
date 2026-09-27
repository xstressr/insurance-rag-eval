# 拆子问题 · decompose_v6

> 自动生成（2026-09-27），`src/decompose_report.py`。评委 j2；成本按列表价，含拆分调用；延迟是每题各次调用耗时之和。

| 集合 | 系统 | 可回答题答对 | 部分正确 | 拒答题 | 忠实 | 美元 / 千题 | 中位延迟（秒） |
|---|---|---|---|---|---|---|---|
| blind_v6 | 单次 RAG | 10/17 | 7 | 2/3 | 20/20 | 1.10 | 13.2 |
| blind_v6 | 拆分 d2 | 14/17 | 3 | 3/3 | 20/20 | 1.54 | 13.8 |
| blind_v6 | Agent | 15/17 | 2 | 3/3 | 19/20 | 3.03 | 16.4 |

## 逐题（评委判定）

| 集合 | 题 | 单次 RAG | 拆分 | Agent | 拆分的检索词 |
|---|---|---|---|---|---|
| blind_v6 | f01 | correct | correct | correct | 恶性肿瘤——重度 疾病定义；恶性肿瘤——轻度 疾病定义 |
| blind_v6 | f02 | correct | correct | correct | 国寿康宁尊享（2024版） 合同内容变更 联系方式 地址变更 |
| blind_v6 | f03 | correct | correct | correct | 太保阿基米德（2025） 投保人 受益人 |
| blind_v6 | f04 | partial | partial | partial | 泰康惠嘉保2026 保险金申请与给付 理赔进度查询 |
| blind_v6 | f05 | correct | correct | correct | 脑中风后遗症 疾病定义；脑中风后遗症 轻度 中度 疾病定义；脑中风后遗症 保险金申请 证明和资料 |
| blind_v6 | f06 | correct | correct | correct | 太保阿基米德（2025） 保险费的交纳 自动扣款；太保阿基米德（2025） 宽限期 未交费 合同效力中止 |
| blind_v6 | f07 | correct | correct | correct | 国寿康宁尊享（2024版） 保险事故通知；国寿康宁尊享（2024版） 保险金申请所需证明和资料；国寿康宁尊享（2024版） 保险金的申请与给付 |
| blind_v6 | f08 | partial | correct | correct | 太保阿基米德 犹豫期 解除合同 现金价值；太保阿基米德 解除合同 现金价值 退还 |
| blind_v6 | f09 | partial | partial | correct | 太保阿基米德2025 肺癌 疾病定义；太保阿基米德2025 保险金申请所需证明和资料；太保阿基米德2025 保险金的申请与给付 |
| blind_v6 | f10 | correct | correct | correct | 泰康惠嘉保2026 合同内容变更 联系方式 变更 |
| blind_v6 | f11 | partial | correct | correct | 国寿康宁尊享2024 保单贷款；国寿康宁尊享2024 保单贷款 申请 操作；国寿康宁尊享2024 保单贷款 利息 |
| blind_v6 | f12 | partial | correct | correct | 国寿康宁尊享2024 轻度疾病定义 轻微脑中风；国寿康宁尊享2024 中度疾病定义 轻微脑中风；国寿康宁尊享2024 保险责任 轻度疾病保险金；国寿康宁尊享2024 保险金申请 给付 |
| blind_v6 | f13 | partial | correct | correct | 泰康惠嘉保2026 受益人 指定；泰康惠嘉保2026 保险金申请与给付 受益人 银行账户 |
| blind_v6 | f14 | correct | correct | correct | 太保 阿基米德2025 互联网保险 法律效力；太保 阿基米德2025 线下保单 法律效力；互联网保险 监管规定 法律效力 |
| blind_v6 | f15 | correct | correct | correct | 阿基米德 保险金申请所需证明和资料；阿基米德 保险金的申请与给付 理赔时效；阿基米德 急性心肌梗死 疾病定义 |
| blind_v6 | f16 | correct | correct | correct | 保险费的交纳与宽限期 宽限期 期限；宽限期 保险责任 出险 赔付 |
| blind_v6 | f17 | partial | partial | partial | 国寿康宁尊享2024版 保险期间 保障期限 |
| blind_v6 | f18 | correct | correct | correct | 阿基米德 保险费的交纳 交费期间；阿基米德 宽限期 保险费的交纳；阿基米德 合同效力中止与恢复；阿基米德 解除合同与现金价值 |
| blind_v6 | f19 | partial | correct | correct | 泰康惠嘉保 健康告知 甲状腺结节；泰康惠嘉保 投保条件 核保 拒保 |
| blind_v6 | f20 | correct | correct | correct | 泰康惠嘉保2026 保险责任 责任免除 保险期间 保险费；阿基米德 保险责任 责任免除 保险期间 保险费；泰康惠嘉保2026 现金价值 解除合同 犹豫期；阿基米德 现金价值 解除合同 犹豫期 |
