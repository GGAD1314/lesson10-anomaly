# -*- coding: utf-8 -*-
"""独立研究：同特征下 Isolation Forest 与 LOF 排名对照，并生成算法笔记。"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SAFE_COLUMNS = ["txn_id", "customer_id", "txn_date", "txn_hour", "amount", "channel", "merchant_type"]
FEATURES = ["amount", "txn_hour", "daily_cnt"]
df = pd.read_csv(ROOT / "data" / "transactions.csv", usecols=SAFE_COLUMNS)
df["daily_cnt"] = df.groupby(["customer_id", "txn_date"])["txn_id"].transform("count")
X = StandardScaler().fit_transform(df[FEATURES])
lof = LocalOutlierFactor(n_neighbors=20).fit(X)
df["lof_score"] = -lof.negative_outlier_factor_
forest = IsolationForest(n_estimators=300, contamination=0.015, random_state=42).fit(X)
df["if_anomaly_score"] = -forest.score_samples(X)
df["lof_top15"] = df.lof_score.rank(method="first", ascending=False) <= 15
df["if_top15"] = df.if_anomaly_score.rank(method="first", ascending=False) <= 15
df["lof_flagship_98_5"] = df.lof_score >= np.percentile(df.lof_score, 98.5)
df["if_label_1_5pct"] = forest.predict(X) == -1
outcols = ["txn_id", "amount", "txn_hour", "daily_cnt", "lof_score", "if_anomaly_score", "lof_top15", "if_top15", "lof_flagship_98_5", "if_label_1_5pct"]
df[outcols].to_csv(ROOT / "generated" / "research_model_comparison.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
overlap15 = int((df.lof_top15 & df.if_top15).sum())
flag_overlap = int((df.lof_flagship_98_5 & df.if_label_1_5pct).sum())
if_only = df.loc[df.if_top15 & ~df.lof_top15, "txn_id"].tolist()
lof_only = df.loc[df.lof_top15 & ~df.if_top15, "txn_id"].tolist()
notes = f"""# 独立研究：异常检测算法与 contamination

姓名：龙羲玥｜学号：20251030226｜专业：金融计算机专业

## 1. Isolation Forest vs LOF

| 维度 | Isolation Forest（IF） | Local Outlier Factor（LOF） |
|---|---|---|
| 核心直觉 | 随机选特征和切分点；越容易被孤立，路径越短、越异常 | 比较点与近邻的局部可达密度；密度相对邻居越低，LOF 越大 |
| 参照尺度 | 多棵随机树的**全局孤立难度** | k 近邻定义的**局部密度** |
| 主要参数 | n_estimators、max_samples、contamination | n_neighbors、distance/metric、contamination |
| 分数方向（本报告转换后） | `-score_samples(X)` 越大越异常 | `-negative_outlier_factor_` 越大越异常 |
| 优势 | 高维、大样本通常更高效；不需要计算全量两两近邻 | 对全局不稀、但相对邻域异常的点更敏感；分数有直观局部解释 |
| 风险 | 随机轴平行切分未必贴合局部流形；小样本排名会波动 | k 选择敏感；异常点抱团可能掩蔽，正常点也可能被淹没 |

**同数据实证。** 两版都只用 amount/txn_hour/daily_cnt，先标准化；IF 设 300 棵树、random_state=42。各取前 15 时重合 **{overlap15}/15**，LOF 独有 {('、'.join(lof_only) or '无')}，IF 独有 {('、'.join(if_only) or '无')}。LOF 的 98.5 分位名单与 IF 的 contamination=0.015 标签名单重合 **{flag_overlap} 笔**（两版名单数可因阈值与并列处理而不同）。这说明强孤岛可被不同机制共同发现，但算法一致不等于业务事实，仍要复核边界点。

## 2. 教材 10.3 CBLOF vs 10.2 LOF

CBLOF（Cluster-Based Local Outlier Factor，He、Xu、Deng，2003）先聚类，再按簇大小划分大簇/小簇，并把**簇大小与距离**合成为异常分数。设点 p 属于簇 Cᵢ：

- 若 Cᵢ 是大簇，分数为 `|Cᵢ| × dist(p, Cᵢ)`；
- 若 Cᵢ 是小簇，分数为 `|Cᵢ| × min dist(p, Cⱼ)`，其中 Cⱼ 取大簇。

原论文用 α（累计样本比例）和 β（相邻簇大小比）确定大/小簇边界；CBLOF 分数不是概率，也没有 LOF≈1 的通用‘正常’含义。

| 对比 | CBLOF | LOF |
|---|---|---|
| 异常依据 | 离本簇/大簇多远，以及所属簇多大 | 与 k 近邻相比，局部密度多低 |
| 前置结构 | 依赖聚类质量、簇数、α/β 和距离 | 不先硬分簇，但强依赖 k 和距离 |
| 适合情形 | 簇结构清晰，希望以簇为单位压缩计算和解释 | 存在不同密度的局部区域，需要细粒度邻域比较 |
| 典型失败 | 正常小簇被整体高分、异常渗入大簇后被簇中心掩盖 | 异常团 ≥k 时掩蔽；边界正常点被拖入时淹没 |

**选择建议：** 本课凌晨大额孤岛两者都可能发现；若要解释‘它和邻居相比有多反常’，LOF 更直接；若业务已有稳定客户分群，CBLOF 更自然。不能把 CBLOF 的‘小簇’机械等同于欺诈簇。

## 3. contamination 到底是什么、怎么设

scikit-learn 文档定义为数据集中异常所占比例，用于在**拟合训练数据时确定分数阈值**。float 范围为 `(0, 0.5]`；IF 的 `'auto'` 沿用原论文阈值，不是从数据自动估计真实欺诈率。LOF 的 contamination 也只影响 `fit_predict/predict` 的 `offset_` 与 -1/1 标签，不改变 `negative_outlier_factor_` 的连续分数；本课直接对连续 LOF 分数取 98.5 分位，是等价的‘复核约 1.5%’业务划线方式，但不依赖 LOF 的 contamination 参数。

**设置顺序：** ①先看每日人工复核容量，例如 1060 笔只能复核约 16 笔，则先设 1.5%；②再用历史独立验证集估计查全率、误报率和资损，不在本课无标签建模阶段偷看 is_fraud 调参；③按成本调节——漏报成本高就适当放宽，客户打扰/人工成本高就收紧；④用时间外样本监控漂移并定期重估。 contamination 是先验工作比例/阈值工具，不是‘模型确信有这么多人欺诈’，也不保证未来批次恰好同比例。

## 4. 本课最终判断

k=20 的 LOF 适合展示本数据的局部孤岛，但 k=5 的掩蔽实验说明参数假设比‘跑通代码’更重要。实际风控应保留连续分数、记录阈值与版本、分层人工复核，并把 IF 作为交叉检查而不是替代证据。

## 参考资料（访问日期：2026-11-20）

1. scikit-learn：LocalOutlierFactor 与 outlier_detection 用户指南，https://scikit-learn.org/stable/modules/outlier_detection#local-outlier-factor
2. scikit-learn：IsolationForest 参数文档，https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
3. He, Xu, Deng (2003), *Discovering Cluster Based Local Outliers*，http://www.dis.uniroma1.it/~sassano/STAGE/Outliers.pdf
4. PyOD：CBLOF 实现与 α/β 参数说明，https://pyod.readthedocs.io/en/latest/_modules/pyod/models/cblof.html

全程未读取 is_fraud；实验明细见 research_model_comparison.csv。
"""
(ROOT / "generated" / "research_notes.md").write_text(notes, encoding="utf-8")
print(f"IF 与 LOF 前 15 重合：{overlap15}/15；旗舰/IF 标签名单重合：{flag_overlap}")
print("已保存 research_model_comparison.csv 和 research_notes.md")
