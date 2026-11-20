# 异常检测算法与 contamination 调研

姓名：龙羲玥｜学号：20251030226｜专业：金融计算机专业

## 1. Isolation Forest 和 LOF

| 比较项 | Isolation Forest（IF） | Local Outlier Factor（LOF） |
|---|---|---|
| 判断方式 | 随机选择特征和切分点，越容易在较少切分后被单独分开，异常分数越高 | 比较样本与 k 个近邻的局部可达密度，相对密度越低，LOF 越大 |
| 参照范围 | 多棵随机树反映的全局孤立难度 | 由 k 近邻确定的局部区域 |
| 主要参数 | n_estimators、max_samples、contamination | n_neighbors、distance/metric、contamination |
| 本实验的分数处理 | `-score_samples(X)`，数值越大越异常 | `-negative_outlier_factor_`，数值越大越异常 |
| 适用特点 | 高维或大样本下通常更快，不必计算全量两两近邻 | 适合找出相对周围邻居异常的点，分数可以从局部密度解释 |
| 可能的问题 | 轴平行的随机切分不一定符合局部数据形状，小样本排名可能波动 | 对 k 较敏感；异常点聚在一起可能互相掩蔽，边界正常点也可能被拖入名单 |

我也在同一批交易上做了对照。两个模型都只使用 amount、txn_hour、daily_cnt，并先标准化；IF 使用 300 棵树，random_state=42。两版前 15 名重合 15/15，LOF 独有名单：无；IF 独有名单：无。LOF 的 98.5 分位名单与 IF 在 contamination=0.015 时的标签名单重合 15 笔。名单数量会受阈值和并列分数处理影响。本例中两种方法找到了同一组突出交易，但这不能代替人工核实，边界交易仍可能误判。

## 2. CBLOF 和 LOF

CBLOF（Cluster-Based Local Outlier Factor，He、Xu、Deng，2003）先聚类，再按簇的大小划分大簇和小簇。它把簇大小和距离放进异常分数。设点 p 属于簇 Cᵢ：

- Cᵢ 是大簇时，分数为 `|Cᵢ| × dist(p, Cᵢ)`；
- Cᵢ 是小簇时，分数为 `|Cᵢ| × min dist(p, Cⱼ)`，其中 Cⱼ 取大簇。

原论文用 α（累计样本比例）和 β（相邻簇大小比）确定大簇与小簇的边界。CBLOF 分数不是概率，也不能像 LOF≈1 那样作一般的正常参考。

| 比较项 | CBLOF | LOF |
|---|---|---|
| 异常依据 | 所属簇的大小，以及点到本簇或大簇的距离 | 点与 k 个近邻之间的局部密度差 |
| 对数据结构的要求 | 依赖聚类质量、簇数、α/β 和距离 | 不先划分簇，但依赖 k 和距离 |
| 较适合的情况 | 数据有较清楚的簇结构，希望按簇解释结果 | 不同区域密度不同，需要逐点比较邻域 |
| 可能的误判 | 正常小簇整体得到高分，或异常点进入大簇后被簇中心掩盖 | 异常点数量达到或超过 k 时发生掩蔽，边界正常点被拖入时发生淹没 |

本课的凌晨大额交易与白天交易分得较开，两种方法都有可能发现它们。若要解释某笔交易与周围交易的差别，LOF 更直接；如果业务上已有稳定的客户分群，CBLOF 更容易接入。不过，CBLOF 的小簇不能直接当成欺诈簇。

## 3. contamination 的含义和设置

scikit-learn 将 contamination 定义为数据集中异常所占的比例，它在拟合训练数据时用于确定分数阈值，float 的取值范围为 `(0, 0.5]`。IF 的 `'auto'` 沿用原论文的阈值，并不会从当前数据自动估计真实欺诈率。对 LOF 来说，contamination 影响 `fit_predict/predict` 使用的 `offset_` 和 -1/1 标签，不改变 `negative_outlier_factor_` 的连续分数。本课直接对连续 LOF 分数取 98.5 分位，相当于把约 1.5% 的交易放进复核名单，但没有使用 LOF 的 contamination 参数来划线。

实际设置时，我会先看每天能人工复核多少笔。例如 1060 笔交易只能复核约 16 笔，就可以先按 1.5% 划线。随后应使用独立的历史验证集估计查全率、误报率和资金损失。本课建模阶段没有读取 is_fraud；实际调参也不能在同一份数据上反复查看标签试参数。漏报代价高时可以适当放宽阈值；客户打扰或人工成本较高时则收紧。数据随时间变化后，还要用时间外样本重新检查比例。所以，contamination 是控制复核比例的阈值参数，不代表模型已经确认相应比例的交易都是欺诈，也不保证下一批数据仍有相同比例。

## 4. 本次实验后的判断

k=20 的 LOF 能找出这份数据里的凌晨大额交易，k=5 的结果则提醒我检查近邻假设。实际使用时，我会保留连续分数和阈值版本，按风险程度安排人工复核，并用 IF 作交叉检查，而不会把任一算法的输出直接当成欺诈结论。

## 参考资料（访问日期：2026-11-20）

1. scikit-learn：LocalOutlierFactor 与 outlier_detection 用户指南，https://scikit-learn.org/stable/modules/outlier_detection#local-outlier-factor
2. scikit-learn：IsolationForest 参数文档，https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
3. He, Xu, Deng (2003), *Discovering Cluster Based Local Outliers*，http://www.dis.uniroma1.it/~sassano/STAGE/Outliers.pdf
4. PyOD：CBLOF 实现与 α/β 参数说明，https://pyod.readthedocs.io/en/latest/_modules/pyod/models/cblof.html

本实验没有读取 is_fraud，逐笔对照结果见 research_model_comparison.csv。
