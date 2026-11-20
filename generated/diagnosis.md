# 诊断：k=5 时凌晨大额点为什么 LOF≈1

## 症状对比
- k=20 第一名 26.40，k=5 第一名 4.63；不是程序报错，而是参数假设失效。

## 近邻构成取证
- 对 k=20 前 5 点，5 近邻中凌晨孤岛成员占比 100.0%；20 近邻中孤岛外主体占比 50.0%。
- 补充核对全部 15 点的 20 近邻：凌晨孤岛成员占 62.7%，孤岛外主体占 37.3%；因此 62.7% 是孤岛成员占比，不是主体占比。
- 完整前 5 点逐点邻居名单见 neighbor_evidence.csv；连线图只展示空间关系，不使用任何标签。

## k 扫描（固定追踪同一批点）
- k=5→1.02；k=8→1.06；k=10→2.88；k=12→5.73；k=15→12.61；k=20→21.45；k=50→21.30；k=200→13.57。
- 纵轴始终是 k=20 前 15 交易在各 k 下的平均分，不是每个版本各自 top15 的平均分。

## scikit-learn 官方依据
> The number k of neighbors considered, (alias parameter n_neighbors) is typically chosen 1) greater than the minimum number of objects a cluster has to contain, so that other objects can be local outliers relative to this cluster, and 2) smaller than the maximum number of close by objects that can potentially be local outliers.
> In practice, such information is generally not available, and taking n_neighbors=20 appears to work well in general.

来源：https://scikit-learn.org/stable/modules/outlier_detection#local-outlier-factor （访问：2026-11-20）。

## 说人话结论
k=5 时，凌晨大额点环顾四周全是同伙：‘我的密度 ÷ 我邻居的密度 ≈ 1’，于是抱团异常被当成正常。
k=20 时，近邻被迫延伸到正常主体，密度对比恢复；这就是掩蔽效应（masking），不是把代码改回 20 就算理解。
