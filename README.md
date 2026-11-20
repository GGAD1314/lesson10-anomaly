# lesson10-anomaly · 第 10 课异常检测

**姓名：龙羲玥｜学号：20251030226｜专业：金融计算机专业**  
课程：《人工智能程序设计与实践》2026 秋｜日期：2026-11-20  
提交仓库：https://github.com/GGAD1314/lesson10-anomaly

本目录是独立、完整的第 10 课作业提交，包含案例复现、两道自主实践、独立研究及选做反欺诈预警台。所有数值均由本仓代码真实运行产生，图表为中文、150 dpi，并标注“教学虚构数据”。

## 作业导航

| 要求 | 完成内容 | 主要证据 |
|---|---|---|
| 案例复现 Step1 | 数据初检、三特征标准化、LOF(k=20)、前 15 名单、名单快照 | `generated/suspicious_list.csv`、`step1_summary.md`、`outputs/01_raw_scatter.png`、`02_lof_result.png` |
| 参数实验与诊断 | k=5 对照、近邻取证、固定名单 k 扫描、官方依据 | `suspicious_list_k5.csv`、`neighbor_evidence.csv`、`diagnosis.md`、`outputs/03_masking_compare.png`、`04_neighbor_evidence.png` |
| 案例复现 Step2 | 98.5 分位阈值、旗舰名单、天平图、三行风控说明及阈值问答 | `flagship_list.csv`、`risk_report.md`、`threshold_sweep.csv`、`outputs/05_threshold_balance.png` |
| 自主实践 1 | 只用 amount 对比三特征，解释名单重合度与适用边界 | `feature_comparison.md`、`suspicious_list_amount_only.csv`、`outputs/06_feature_compare.png` |
| 自主实践 2 | bank_customers 标准化、KMeans、LOF，报告离群客户簇归属 | `bank_lof_report.md`、`bank_lof_outliers.csv`、`bank_cluster_profile.csv`、`outputs/07_bank_lof_clusters.png` |
| 独立研究 | IF vs LOF 实证；CBLOF vs LOF；contamination 含义、设置与成本 | `research_notes.md`、`research_model_comparison.csv` |
| 选做预警台 | 单文件零依赖原生 SVG，阈值滑杆联动散点/表格/KPI/报告 | `webapp/index.html`（双击可打开） |
| 过程材料 | 检查点、问题与修复、任务自查、AI 使用与验证 | `progress.md`、`AI使用记录.md` |

## 核心结果

- 原始交易数据 **1060×8**；分析视图只读取七个非标注字段，无缺失。
- Step1 前 15 全部位于凌晨 **1—4 点**，金额 **16,553.09—43,690.22 元**；LOF 第 1 → 第 15 为 **26.40 → 17.62**。
- k=5 与 k=20 前 15 重合 **0/15**；k=5 第一名 **4.63**，名单转为白天交易，金额中位 **1,239 元**。
- 固定追踪 k=20 前 15 的同一批交易，k 扫描平均 LOF：**1.02 / 1.06 / 2.88 / 5.73 / 12.61 / 21.45 / 21.30 / 13.57**（k=5/8/10/12/15/20/50/200）。这不是“每版各自 top15 均值”的口径。
- 98.5 分位阈值 **2.895355**，旗舰名单 **16 笔**，金额合计 **429,286.56 元**；15 笔凌晨大额 + T0133（16 点/柜台/超市/2,926.63 元）。
- 97 分位 **32 笔**，99.5 分位 **6 笔**。模型冻结后的独立揭盲：98.5 分位检中 **15/15**、误报 **1**；99.5 分位检中 **6**、漏报 **9**。
- 金额单特征与三特征前 15 重合 **15/15**，但顺序不完全相同；只说明本数据金额信号强，不能推广为其他特征无用。
- bank_customers 在明确的 5 簇/20 近邻/5% 口径下有 **10** 名 LOF 离群客户：**簇 0 为 9 人，簇 3 为 1 人**，其余簇为 0。
- IF 与 LOF 同特征前 15 重合 **15/15**；算法一致仍不等于业务定罪。

## 复现方法

环境：Windows、Python 3；依赖见 `requirements.txt`。

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

脚本使用相对于项目根目录的路径，可从仓库根目录直接运行。01—08 是建模/分析阶段；09 是模型完成后的独立揭盲评估，并把真实分数内嵌进看板。看板本身不发网络请求、不加载外部库，直接双击 `webapp/index.html` 即可。

## 数据红线与来源

1. `data/transactions.csv` 来自课堂 `lesson10-课堂演示包.zip`，原样保留。建模和分析脚本 **01—08 均用 `usecols` 排除 is_fraud**，不把它作为特征、标签或调参依据。
2. 只有 `09_post_evaluation_and_dashboard.py` 在上游名单、阈值和研究冻结后读取该列，用于课程要求的揭盲检中数/漏报数；评估结果不回写模型。
3. `data/bank_customers.csv` 复用课程第 7 课公开仓库数据，原样保留；其 `is_default/risk_level` 也未读取。自主实践 2 只使用七个数值画像字段，题面未指定簇数，因此明确声明 KMeans=5 的假设。
4. 两份原始数据的 SHA256 见 `data/SHA256SUMS.txt`；所有输出均为教学虚构数据分析，不代表真实客户或真实欺诈率。

## 对演示材料的必要修正

- LOF 排序图将升序散点的 x 坐标与前 15 高分点严格对齐，避免“左侧圈点、右侧才是高分”的错位。
- k 扫描固定追踪同一批 k=20 前 15 交易，不在每个 k 下更换观测对象。
- k=5 第一名按实跑结果写 **4.63**；全部 15 点的 20 近邻中，凌晨孤岛成员为 **62.7%**、孤岛外主体为 **37.3%**，不混用两个比例。
- 修复看板演示版“只改透明度/半径、不变红”的问题；本版每个滑杆位置都按 `lof_score >= threshold` 重算红色集合、表格、金额、检中数和报告。
- 统一 UTF-8、Windows 中文字体回退和 150 dpi 输出。

## 目录

```text
lesson10-anomaly/
├── data/                 # 两份原始 CSV + SHA256
├── prompts/              # 课堂五条原始提示词
├── generated/            # 9 个脚本、名单、分数、报告、评估与看板数据
├── outputs/              # 7 张中文分析图（150 dpi）
├── webapp/index.html     # 单文件零依赖预警台
├── README.md
├── progress.md
├── AI使用记录.md
└── requirements.txt
```
