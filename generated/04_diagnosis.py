# -*- coding: utf-8 -*-
"""近邻构成取证与 k 扫描；k 扫描固定追踪 k=20 前 15 的同一批交易。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SAFE_COLUMNS = ["txn_id", "customer_id", "txn_date", "txn_hour", "amount", "channel", "merchant_type"]
FEATURES = ["amount", "txn_hour", "daily_cnt"]
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Songti SC", "STHeiti", "PingFang SC"]
matplotlib.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(ROOT / "data" / "transactions.csv", usecols=SAFE_COLUMNS)
df["daily_cnt"] = df.groupby(["customer_id", "txn_date"])["txn_id"].transform("count")
X = StandardScaler().fit_transform(df[FEATURES])
scores20 = -LocalOutlierFactor(n_neighbors=20).fit(X).negative_outlier_factor_
scores5 = -LocalOutlierFactor(n_neighbors=5).fit(X).negative_outlier_factor_
order20 = np.argsort(scores20)[::-1]
probe_idx = order20[:5]
tracked_idx = order20[:15]

print("k=20 前 5：")
print(df.iloc[order20[:5]].assign(lof_score=scores20[order20[:5]])[["txn_id", "amount", "txn_hour", "lof_score"]].round({"lof_score": 2}).to_string(index=False))
print("k=5 前 5：")
order5 = np.argsort(scores5)[::-1]
print(df.iloc[order5[:5]].assign(lof_score=scores5[order5[:5]])[["txn_id", "amount", "txn_hour", "lof_score"]].round({"lof_score": 2}).to_string(index=False))

neighbor_rows = []
neighbor_indices = {}
for k in (5, 20):
    _, indices = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X[probe_idx])
    neighbor_indices[k] = indices
    for rank, (pidx, row) in enumerate(zip(probe_idx, indices), start=1):
        for neighbor_rank, nidx in enumerate(row[1:], start=1):
            neighbor_rows.append({
                "probe_rank": rank, "probe_txn_id": df.iloc[pidx].txn_id, "k": k,
                "neighbor_rank": neighbor_rank, "neighbor_txn_id": df.iloc[nidx].txn_id,
                "neighbor_txn_hour": df.iloc[nidx].txn_hour, "neighbor_amount": df.iloc[nidx].amount,
            })
neighbor_df = pd.DataFrame(neighbor_rows)
neighbor_df.to_csv(ROOT / "generated" / "neighbor_evidence.csv", index=False, encoding="utf-8-sig")

ks = [5, 8, 10, 12, 15, 20, 50, 200]
avg = []
for k in ks:
    scores = -LocalOutlierFactor(n_neighbors=k).fit(X).negative_outlier_factor_
    avg.append(scores[tracked_idx].mean())
pd.DataFrame({"n_neighbors": ks, "tracked_k20_top15_mean_lof": avg}).to_csv(
    ROOT / "generated" / "k_sweep.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

night_ids = set(df.loc[df.txn_hour.between(1, 4), "txn_id"])
same5 = neighbor_df[(neighbor_df.k == 5) & (neighbor_df.neighbor_txn_id.isin(night_ids))].shape[0] / len(neighbor_df[neighbor_df.k == 5])
main20 = neighbor_df[(neighbor_df.k == 20) & (~neighbor_df.neighbor_txn_id.isin(night_ids))].shape[0] / len(neighbor_df[neighbor_df.k == 20])
_, tracked_neighbors = NearestNeighbors(n_neighbors=21).fit(X).kneighbors(X[tracked_idx])
tracked_night_share = np.mean([df.iloc[row[1:]].txn_hour.between(1, 4).mean() for row in tracked_neighbors])
quote = ("The number k of neighbors considered, (alias parameter n_neighbors) is typically chosen 1) greater than "
         "the minimum number of objects a cluster has to contain, so that other objects can be local outliers relative "
         "to this cluster, and 2) smaller than the maximum number of close by objects that can potentially be local outliers.")
lines = [
    "# 诊断：k=5 时凌晨大额点为什么 LOF≈1", "",
    "## 症状对比",
    f"- k=20 第一名 {scores20.max():.2f}，k=5 第一名 {scores5.max():.2f}；不是程序报错，而是参数假设失效。", "",
    "## 近邻构成取证",
    f"- 对 k=20 前 5 点，5 近邻中凌晨孤岛成员占比 {same5:.1%}；20 近邻中孤岛外主体占比 {main20:.1%}。",
    f"- 补充核对全部 15 点的 20 近邻：凌晨孤岛成员占 {tracked_night_share:.1%}，孤岛外主体占 {1-tracked_night_share:.1%}；因此 62.7% 是孤岛成员占比，不是主体占比。",
    "- 完整前 5 点逐点邻居名单见 neighbor_evidence.csv；连线图只展示空间关系，不使用任何标签。", "",
    "## k 扫描（固定追踪同一批点）",
    "- " + "；".join(f"k={k}→{v:.2f}" for k, v in zip(ks, avg)) + "。",
    "- 纵轴始终是 k=20 前 15 交易在各 k 下的平均分，不是每个版本各自 top15 的平均分。", "",
    "## scikit-learn 官方依据",
    f"> {quote}",
    "> In practice, such information is generally not available, and taking n_neighbors=20 appears to work well in general.", "",
    "来源：https://scikit-learn.org/stable/modules/outlier_detection#local-outlier-factor （访问：2026-11-20）。", "",
    "## 说人话结论",
    "k=5 时，凌晨大额点环顾四周全是同伙：‘我的密度 ÷ 我邻居的密度 ≈ 1’，于是抱团异常被当成正常。",
    "k=20 时，近邻被迫延伸到正常主体，密度对比恢复；这就是掩蔽效应（masking），不是把代码改回 20 就算理解。",
]
(ROOT / "generated" / "diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
for ax, k in [(axes[0], 5), (axes[1], 20)]:
    ax.scatter(df.txn_hour, df.amount, s=10, alpha=0.3, color="#9db8d9", edgecolors="none")
    for row, pidx in zip(neighbor_indices[k], probe_idx):
        nb = df.iloc[row[1:]]
        ax.scatter(nb.txn_hour, nb.amount, s=26, color="#2ca02c", alpha=0.8, zorder=3)
        for _, r in nb.iterrows():
            ax.plot([df.iloc[pidx].txn_hour, r.txn_hour], [df.iloc[pidx].amount, r.amount], color="#2ca02c", linewidth=0.6, alpha=0.55, zorder=2)
    pts = df.iloc[probe_idx]
    ax.scatter(pts.txn_hour, pts.amount, s=110, facecolors="none", edgecolors="#d62728", linewidths=1.8, zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel("交易小时（点）")
    ax.set_title(f"k={k}：名单点的邻居是谁")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("交易金额（元，对数轴）")
axes[2].plot(ks, avg, "o-", color="#d62728")
axes[2].set_xscale("log")
axes[2].set_xticks(ks)
axes[2].set_xticklabels([str(k) for k in ks])
axes[2].set_xlabel("n_neighbors（对数轴）")
axes[2].set_ylabel("固定 k=20 前 15 点的平均 LOF")
axes[2].set_title("k 扫描：同一批点的分数变化")
axes[2].grid(alpha=0.3)
for ax in axes:
    ax.annotate("教学虚构数据", xy=(0.99, 0.02), xycoords="axes fraction", ha="right", fontsize=9, color="gray")
fig.tight_layout()
fig.savefig(ROOT / "outputs" / "04_neighbor_evidence.png", dpi=150)
plt.close(fig)
print("k 扫描固定名单平均分：", [round(v, 2) for v in avg])
print("已保存 diagnosis.md、neighbor_evidence.csv、k_sweep.csv 和 04_neighbor_evidence.png")
