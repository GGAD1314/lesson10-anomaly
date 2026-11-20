# -*- coding: utf-8 -*-
"""自主实践 2：bank_customers 数值画像 + KMeans(5) + LOF(20, 5%)。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["age", "account_age_months", "product_count", "balance", "avg_monthly_spend", "loan_flag", "overdue_count"]
USECOLS = ["customer_id"] + FEATURES
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Songti SC", "STHeiti", "PingFang SC"]
matplotlib.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(ROOT / "data" / "bank_customers.csv", usecols=USECOLS)
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(df[FEATURES])
X = StandardScaler().fit_transform(X_imputed)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof_labels = lof.fit_predict(X)
df["cluster"] = kmeans.labels_
df["lof_score"] = -lof.negative_outlier_factor_
df["is_lof_outlier"] = lof_labels == -1
out = df[["customer_id"] + FEATURES + ["cluster", "lof_score", "is_lof_outlier"]]
out.to_csv(ROOT / "generated" / "bank_lof_clusters.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
outliers = out[out.is_lof_outlier].sort_values("lof_score", ascending=False)
outliers.to_csv(ROOT / "generated" / "bank_lof_outliers.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
profile = df.groupby("cluster").agg(customer_count=("customer_id", "size"), outlier_count=("is_lof_outlier", "sum"), mean_lof=("lof_score", "mean"), mean_balance=("balance", "mean"), mean_monthly_spend=("avg_monthly_spend", "mean"), mean_overdue=("overdue_count", "mean")).reset_index()
profile["outlier_share"] = profile.outlier_count / len(outliers)
profile.to_csv(ROOT / "generated" / "bank_cluster_profile.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
cluster_text = "；".join(f"簇 {int(r.cluster)}：{int(r.outlier_count)} 人（占离群者 {r.outlier_share:.1%}，簇内 {int(r.outlier_count)}/{int(r.customer_count)}）" for _, r in profile.iterrows())
lines = [
    "# 自主实践 2：bank_customers 的 LOF 离群客户落在哪些聚类簇", "",
    "姓名：龙羲玥｜学号：20251030226｜专业：金融计算机专业", "",
    "## 复用口径与必要假设",
    "复用第 9 课的‘中位数填补 → StandardScaler → KMeans’思路；题面未给簇数，本报告明确设 KMeans(n_clusters=5, random_state=42, n_init=10)。",
    "聚类与 LOF 使用同一套 7 个数值特征；LOF(n_neighbors=20, contamination=0.05)，即训练分数阈值约 5%，得到 10 名离群客户。",
    "customer_id 仅作标识；gender/city 不纳入本次数值画像；is_default/risk_level 是监督标签，未读取、未用于选参或建模。", "",
    "## 结果：离群客户的簇归属",
    cluster_text + "。", "",
    "## 解释",
    "‘离群’由局部密度决定，‘簇’由全局质心划分决定，两者不是同一概念；离群者可以集中在一个边缘小簇，也可能分散在多个簇。",
    "因此应同时看 outlier_count（离群者去了哪里）和簇内比例（该簇有多反常），不能只看簇的大小或把整簇直接判为高风险。", "",
    "## 局限",
    "结果依赖特征集合、5 簇设定、20 近邻和 5% 复核比例；这是无监督筛查线索，不是违约结论。完整逐人结果见 bank_lof_clusters.csv。",
]
(ROOT / "generated" / "bank_lof_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

xy = PCA(n_components=2, random_state=42).fit_transform(X)
fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
for c in sorted(df.cluster.unique()):
    mask = df.cluster == c
    axes[0].scatter(xy[mask, 0], xy[mask, 1], s=25, alpha=0.65, label=f"簇 {c}")
mask = df.is_lof_outlier
axes[0].scatter(xy[mask, 0], xy[mask, 1], s=115, facecolors="none", edgecolors="#d62728", linewidths=1.6, label="LOF 离群")
for _, r in df[mask].iterrows():
    i = df.index.get_loc(r.name)
    axes[0].annotate(r.customer_id, (xy[i, 0], xy[i, 1]), xytext=(4, 4), textcoords="offset points", fontsize=8)
axes[0].set_title("客户二维投影与 LOF 离群点（PCA 仅用于画图）")
axes[0].set_xlabel("主成分 1")
axes[0].set_ylabel("主成分 2")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.25)
axes[1].bar(profile.cluster.astype(str), profile.customer_count, color="#9db8d9", label="簇人数")
axes[1].bar(profile.cluster.astype(str), profile.outlier_count, color="#d64550", label="其中 LOF 离群")
for i, r in profile.iterrows():
    axes[1].text(i, r.outlier_count + 0.25, str(int(r.outlier_count)), ha="center", color="#a92e38")
axes[1].set_title("离群客户的聚类簇分布")
axes[1].set_xlabel("KMeans 簇编号")
axes[1].set_ylabel("客户数")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.25)
for ax in axes:
    ax.annotate("教学虚构数据", xy=(0.99, 0.02), xycoords="axes fraction", ha="right", fontsize=9, color="gray")
fig.tight_layout()
fig.savefig(ROOT / "outputs" / "07_bank_lof_clusters.png", dpi=150)
plt.close(fig)
print(profile.round({"mean_lof": 2, "outlier_share": 2}).to_string(index=False))
print("已保存自主实践 2 的逐人名单、簇画像、报告和图")
