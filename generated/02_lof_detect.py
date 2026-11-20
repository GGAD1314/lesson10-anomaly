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
hours = "、".join(str(int(h)) for h in sorted(top15.txn_hour.unique()))
channels = "、".join(f"{name} {count} 笔" for name, count in top15.channel.value_counts().items())
merchants = "、".join(f"{name} {count} 笔" for name, count in top15.merchant_type.value_counts().items())
lines = [
    "# Step1 可疑名单结果", "",
    "姓名：龙羲玥｜学号：20251030226｜专业：金融计算机专业", "",
    f"- 按 LOF 分数从高到低取前 15 笔，第 1 名为 {top15.lof_score.max():.2f}，第 15 名为 {top15.lof_score.min():.2f}。",
    f"- 这些交易都发生在凌晨 {hours} 点。",
    f"- 金额范围为 {top15.amount.min():,.2f}—{top15.amount.max():,.2f} 元，中位数为 {top15.amount.median():,.2f} 元；全部交易的中位数只有 {df.amount.median():,.2f} 元。",
    f"- 渠道分布：{channels}；商户类型：{merchants}。",
    "- 在小时—金额散点图上，这 15 笔交易与白天交易分开。它们的局部密度也明显低于各自近邻，因此 LOF 分数较高。",
    "- 计算时只使用 amount、txn_hour、daily_cnt，先做 StandardScaler，再计算 LOF(k=20)，没有读取 is_fraud。",
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
