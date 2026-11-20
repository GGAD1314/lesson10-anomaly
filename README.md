# 第 10 课异常检测作业

姓名：龙羲玥｜学号：20251030226｜专业：金融计算机专业
课程：《人工智能程序设计与实践》2026 秋｜日期：2026-11-20
提交仓库：https://github.com/GGAD1314/lesson10-anomaly

这个目录包含案例复现、k=5 诊断、两道自主实践、算法调研和选做预警台。报告中的数字都来自本目录脚本的实际运行结果。图表使用中文，分辨率为 150 dpi，并标有“教学虚构数据”。

## 作业导航

| 题目 | 我完成的内容 | 对应文件 |
|---|---|---|
| 案例 Step1 | 检查数据，标准化三个特征，用 LOF(k=20) 取前 15 笔可疑交易 | `generated/suspicious_list.csv`、`step1_summary.md`、`outputs/01_raw_scatter.png`、`02_lof_result.png` |
| k=5 实验与诊断 | 比较 k=5 和 k=20，检查近邻构成，并固定同一批交易扫描 k | `suspicious_list_k5.csv`、`neighbor_evidence.csv`、`diagnosis.md`、`outputs/03_masking_compare.png`、`04_neighbor_evidence.png` |
| 案例 Step2 | 计算 98.5 分位阈值，导出预警名单，画阈值图并写处理建议 | `flagship_list.csv`、`risk_report.md`、`threshold_sweep.csv`、`outputs/05_threshold_balance.png` |
| 自主实践 1 | 只用 amount 重跑 LOF，与三特征版本比较 | `feature_comparison.md`、`suspicious_list_amount_only.csv`、`outputs/06_feature_compare.png` |
| 自主实践 2 | 对 bank_customers 做标准化、KMeans 和 LOF，统计离群客户的簇归属 | `bank_lof_report.md`、`bank_lof_outliers.csv`、`bank_cluster_profile.csv`、`outputs/07_bank_lof_clusters.png` |
| 独立研究 | 对比 IF 与 LOF，整理 CBLOF 原理，说明 contamination 的设置方法 | `research_notes.md`、`research_model_comparison.csv` |
| 选做预警台 | 用原生 SVG 制作单文件看板，滑杆可联动散点、表格、指标和报告 | `webapp/index.html`，双击即可打开 |
| 过程记录 | 记录检查点、遇到的问题、自查结果和 AI 使用情况 | `progress.md`、`AI使用记录.md` |

## 实验结果

- transactions.csv 有 1060 行、8 列。建模时只读取 7 个非标注字段，没有缺失值。
- Step1 前 15 笔都在凌晨 1—4 点，金额为 16,553.09—43,690.22 元。LOF 第 1 名为 26.40，第 15 名为 17.62。
- k=5 与 k=20 的前 15 名没有重合，k=5 最高分为 4.63。k=5 名单主要是白天交易，金额中位数为 1,239 元。
- 固定观察 k=20 的前 15 笔交易，k=5/8/10/12/15/20/50/200 时的平均 LOF 分别为 1.02、1.06、2.88、5.73、12.61、21.45、21.30、13.57。这里没有在每个 k 下重新选择前 15 名。
- 98.5 分位阈值为 2.895355，对应 16 笔交易，金额合计 429,286.56 元。其中 15 笔是凌晨大额交易，另一笔是 T0133（16 点、柜台、超市、2,926.63 元）。
- 97 分位对应 32 笔，99.5 分位对应 6 笔。模型冻结后读取标签核验：98.5 分位检中 15/15，误报 1 笔；99.5 分位检中 6 笔，漏报 9 笔。
- 只用 amount 与使用三个特征得到的前 15 名完全重合，但顺序不同。这个结果只说明本数据的金额差异较大，不能说明其他特征在别的数据上没有用。
- bank_customers 在 5 簇、20 近邻、5% 离群比例的设置下得到 10 名离群客户，其中簇 0 有 9 人，簇 3 有 1 人，其余簇没有。
- IF 与 LOF 使用相同特征时，前 15 名重合 15/15。两个模型名单一致，也不等于交易已经被确认为欺诈。

## 运行方法

环境为 Windows、Python 3，依赖列在 `requirements.txt`。在仓库根目录依次运行：

```powershell
py -3 -m pip install -r requirements.txt
py -3 generated/01_load_and_viz.py
py -3 generated/02_lof_detect.py
py -3 generated/03_masking_experiment.py
py -3 generated/04_diagnosis.py
py -3 generated/05_threshold_report.py
py -3 generated/06_feature_comparison.py
py -3 generated/07_bank_lof_clusters.py
py -3 generated/08_independent_research.py
py -3 generated/09_post_evaluation_and_dashboard.py
```

脚本中的路径都相对于项目根目录。01—08 负责建模和分析，09 在这些结果确定后单独合并标签，计算检中数、漏报数，并把分数写入看板。看板不加载外部库，也不发网络请求。

## 数据使用说明

1. `data/transactions.csv` 来自课堂演示包，原文件没有修改。01—08 都通过 `usecols` 排除 is_fraud，没有把它用于特征、选参或阈值选择。
2. 只有 `09_post_evaluation_and_dashboard.py` 在上游结果冻结后读取 is_fraud，用于课程要求的揭盲核验。核验结果不会回写到建模脚本。
3. `data/bank_customers.csv` 复用课程第 7 课公开仓库的数据，也保持原样。is_default、risk_level 没有读取。自主实践 2 只使用 7 个数值画像字段；题目未给簇数，所以报告中明确写了 KMeans=5。
4. 两份原始数据的 SHA256 记录在 `data/SHA256SUMS.txt`。所有结论只针对教学虚构数据，不代表真实客户或真实欺诈率。

## 对演示代码做的修正

- 修正 LOF 排序图的横坐标，让高分圈点与升序散点的位置一致。
- 扫描 k 时始终观察 k=20 的前 15 笔交易，避免中途更换观察对象。
- 按实际输出写 k=5 最高分 4.63。重新统计全部 15 笔的 20 近邻后，凌晨交易占 62.7%，白天交易占 37.3%。
- 看板在每次移动滑杆后重新计算 `lof_score >= threshold` 的交易集合同步更新颜色、表格、金额、检中数、漏报数和建议，不再只改变点的透明度或半径。
- 统一使用 UTF-8、Windows 中文字体回退和 150 dpi 图片输出。

## 文件结构

```text
lesson10-anomaly/
├── data/                 # 原始 CSV 与 SHA256
├── prompts/              # 课堂提供的五条原始提示词
├── generated/            # 9 个脚本、名单、分数、报告和看板数据
├── outputs/              # 7 张中文分析图
├── webapp/index.html     # 单文件预警台
├── README.md
├── progress.md
├── AI使用记录.md
└── requirements.txt
```
