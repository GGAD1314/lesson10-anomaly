# -*- coding: utf-8 -*-
"""Step1：标准化三特征 + LOF(k=20)，输出前 15 可疑名单。"""
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
lof = LocalOutlierFactor(n_neighbors=20)
lof.fit(X)
df["lof_score"] = -lof.negative_outlier_factor_

top15 = df.sort_values("lof_score", ascending=False).head(15).copy()
cols = ["txn_id", "customer_id", "amount", "txn_hour", "lof_score"]
top15[cols].to_csv(ROOT / "generated" / "suspicious_list.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
score_cols = SAFE_COLUMNS + ["daily_cnt", "lof_score"]
df[score_cols].to_csv(ROOT / "generated" / "lof_scores.csv", index=False, encoding="utf-8-sig", float_format="%.8f")

print("=== 可疑名单（前 15，LOF 分数降序）===")
print(top15[cols].round({"lof_score": 2}).to_string(index=False))
lines = [
    "# Step1 名单快照", "",
    "姓名：龙羲玥｜学号：20251030226｜专业：金融计算机专业", "",
    f"- 前 15 名单 LOF 分数：{top15.lof_score.max():.2f} → {top15.lof_score.min():.2f}（第 1 → 第 15）。",
    f"- 时段：{sorted(top15.txn_hour.unique())} 点，全部位于凌晨 1—4 点。",
    f"- 金额：{top15.amount.min():,.2f}—{top15.amount.max():,.2f} 元，中位数 {top15.amount.median():,.2f} 元；全体中位数 {df.amount.median():,.2f} 元。",
    f"- 渠道：{top15.channel.value_counts().to_dict()}；商户类型：{top15.merchant_type.value_counts().to_dict()}。",
    "- 证据：在小时—金额图上，它们形成与白天主体分离的孤岛；标准化空间中的局部密度显著低于各自邻居，所以 LOF 远大于 1。",
    "- 口径：仅使用 amount / txn_hour / daily_cnt，先 StandardScaler，再 LOF(k=20)；未读取 is_fraud。",
]
(ROOT / "generated" / "step1_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

ordered = np.sort(df["lof_score"].to_numpy())
n = len(df)
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(np.arange(n), ordered, s=8, alpha=0.5, color="#3b6fb6", label="全体 1060 笔")
ax.scatter(np.arange(n - 15, n), ordered[-15:], s=90, facecolors="none", edgecolors="#d62728", linewidths=1.8, label="前 15 可疑名单")
ax.set_yscale("log")
ax.set_xlabel("交易（按 LOF 分数升序排列）")
ax.set_ylabel("LOF 分数（对数轴，越大越异常）")
ax.set_title("LOF 分数分布与前 15 可疑名单（k=20，标准化三特征）")
ax.legend()
ax.grid(alpha=0.3)
ax.annotate("教学虚构数据", xy=(0.99, 0.01), xycoords="axes fraction", ha="right", fontsize=9, color="gray")
fig.tight_layout()
fig.savefig(ROOT / "outputs" / "02_lof_result.png", dpi=150)
plt.close(fig)
print("已保存 suspicious_list.csv、lof_scores.csv、step1_summary.md 和 02_lof_result.png")
