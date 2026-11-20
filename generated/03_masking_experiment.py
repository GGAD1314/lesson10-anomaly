# -*- coding: utf-8 -*-
"""故意实验：只把 n_neighbors 从 20 改为 5，观察掩蔽效应。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SAFE_COLUMNS = ["txn_id", "customer_id", "txn_date", "txn_hour", "amount", "channel", "merchant_type"]
FEATURES = ["amount", "txn_hour", "daily_cnt"]
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Songti SC", "STHeiti", "PingFang SC"]
matplotlib.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(ROOT / "data" / "transactions.csv", usecols=SAFE_COLUMNS)
df["daily_cnt"] = df.groupby(["customer_id", "txn_date"])["txn_id"].transform("count")
X = StandardScaler().fit_transform(df[FEATURES])

def detect(k):
    model = LocalOutlierFactor(n_neighbors=k).fit(X)
    scores = -model.negative_outlier_factor_
    idx = np.argsort(scores)[::-1][:15]
    out = df.iloc[idx].copy()
    out["lof_score"] = scores[idx]
    return out, scores

top20, scores20 = detect(20)
top5, scores5 = detect(5)
cols = ["txn_id", "customer_id", "amount", "txn_hour", "lof_score"]
top5[cols].to_csv(ROOT / "generated" / "suspicious_list_k5.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
overlap = len(set(top20.txn_id) & set(top5.txn_id))
print(f"新旧名单重合：{overlap}/15")
print(f"k=20 第一名：{scores20.max():.2f}；k=5 第一名：{scores5.max():.2f}")
print("k=5 名单时段分布：", top5.txn_hour.value_counts().sort_index().to_dict())
print(f"k=5 名单金额中位：{top5.amount.median():.0f} 元")

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
for ax, top, k in [(axes[0], top20, 20), (axes[1], top5, 5)]:
    ax.scatter(df.txn_hour, df.amount, s=10, alpha=0.35, color="#9db8d9", edgecolors="none")
    ax.scatter(top.txn_hour, top.amount, s=110, facecolors="none", edgecolors="#d62728", linewidths=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("交易小时（点）")
    ax.set_title(f"k={k} 的前 15 名单")
    ax.grid(alpha=0.3)
    ax.annotate("教学虚构数据", xy=(0.99, 0.02), xycoords="axes fraction", ha="right", fontsize=9, color="gray")
axes[0].set_ylabel("交易金额（元，对数轴）")
fig.suptitle(f"只改 n_neighbors：20 → 5，名单重合 {overlap}/15", fontsize=14)
fig.tight_layout()
fig.savefig(ROOT / "outputs" / "03_masking_compare.png", dpi=150)
plt.close(fig)
print("已保存 suspicious_list_k5.csv 和 03_masking_compare.png")
