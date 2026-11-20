# k=5 时凌晨大额交易的 LOF 为什么接近 1

## 实验现象
k=20 时最高分为 26.40，改成 k=5 后最高分只有 4.63，而且高分名单也变了。问题出在近邻数量，不是程序没有跑通。

## 查看近邻构成
- 先取 k=20 名单的前 5 笔。它们的 5 个近邻全部是凌晨交易，占 100.0%；把近邻数增加到 20 后，白天交易占 50.0%。
- 再核对前 15 笔的全部 20 近邻，凌晨交易占 62.7%，白天交易占 37.3%。所以这里的 62.7% 指凌晨交易，不是白天交易。
- 前 5 笔交易的逐个近邻已保存到 neighbor_evidence.csv。图中连线只表示特征空间里的近邻关系，没有使用标签。

## 固定同一批交易扫描 k
- k=5 时为 1.02；k=8 时为 1.06；k=10 时为 2.88；k=12 时为 5.73；k=15 时为 12.61；k=20 时为 21.45；k=50 时为 21.30；k=200 时为 13.57。
- 每个数都是 k=20 前 15 笔交易在相应 k 下的平均 LOF。扫描过程中没有更换观察对象，因此不能把它理解成各版本前 15 名的平均分。

## scikit-learn 文档中的说明
> The number k of neighbors considered, (alias parameter n_neighbors) is typically chosen 1) greater than the minimum number of objects a cluster has to contain, so that other objects can be local outliers relative to this cluster, and 2) smaller than the maximum number of close by objects that can potentially be local outliers.
> In practice, such information is generally not available, and taking n_neighbors=20 appears to work well in general.

来源：https://scikit-learn.org/stable/modules/outlier_detection#local-outlier-factor （访问：2026-11-20）。

## 原因
只看 5 个近邻时，这些凌晨大额交易的邻居仍来自同一小组，彼此密度接近。“我的密度 ÷ 我邻居的密度”因而接近 1，算法不会把它们单独挑出来。
k 增加到 20 后，近邻中开始出现白天普通交易，局部密度差异才显现出来。这就是本例的掩蔽效应（masking）。
