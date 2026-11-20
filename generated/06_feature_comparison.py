# -*- coding: utf-8 -*-
"""自主实践 1：只用 amount 与三特征模型比较，仍不使用 is_fraud。"""
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
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Songti SC", "STHeiti", "PingFang SC"]
matplotlib.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(ROOT / "data" / "transactions.csv", usecols=SAFE_COLUMNS)
df["daily_cnt"] = df.groupby(["customer_id", "txn_date"])["txn_id"].transform("count")

def detect(features):
    X = StandardScaler().fit_transform(df[features])
    scores = -LocalOutlierFactor(n_neighbors=20).fit(X).negative_outlier_factor_
    idx = np.argsort(scores)[::-1][:15]
    top = df.iloc[idx].copy()
    top["lof_score"] = scores[idx]
    return top, scores

top3, scores3 = detect(["amount", "txn_hour", "daily_cnt"])
top_amount, scores_amount = detect(["amount"])
cols = ["txn_id", "customer_id", "amount", "txn_hour", "lof_score"]
top_amount[cols].to_csv(ROOT / "generated" / "suspicious_list_amount_only.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
comparison = pd.DataFrame({
    "txn_id": df.txn_id, "amount": df.amount, "txn_hour": df.txn_hour,
    "three_feature_lof": scores3, "amount_only_lof": scores_amount,
})
comparison.to_csv(ROOT / "generated" / "feature_score_comparison.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
overlap = len(set(top3.txn_id) & set(top_amount.txn_id))
same_order = top3.txn_id.tolist() == top_amount.txn_id.tolist()
rest = df[~df.txn_id.isin(top3.txn_id)]
lines = [
    "# 自主实践 1：只用 amount 与三特征模型对比", "",
    "姓名：龙羲玥｜学号：20251030226｜专业：金融计算机专业", "",
    "## 实验口径",
    "两版都用 StandardScaler + LOF(n_neighbors=20)，各取分数前 15；唯一差别是特征集合。未读取 is_fraud。", "",
    "## 结果",
    f"- 前 15 名单重合 **{overlap}/15（{overlap/15:.1%}）**；名次是否完全一致：{'是' if same_order else '否'}。",
    f"- 三特征版第 1/第 15 分数：{top3.lof_score.max():.2f}/{top3.lof_score.min():.2f}；单金额版：{top_amount.lof_score.max():.2f}/{top_amount.lof_score.min():.2f}。",
    f"- 三特征前 15 的 daily_cnt 均值 {top3.daily_cnt.mean():.2f}，其余交易均值 {rest.daily_cnt.mean():.2f}，该特征在本数据上区分度弱。", "",
    "## 为什么高度重合",
    "凌晨大额孤岛在 amount 一维上已经与正常交易拉开极大距离，即使去掉小时和当日笔数，局部密度对比仍足以让同批交易居前。",
    "txn_hour 在此主要提供与金额一致的辅助信号；daily_cnt 变化小，对近邻距离的额外贡献有限，所以名单高度稳定。", "",
    "## 能否据此删掉特征",
    "不能推广为‘金额永远足够’。本教学数据的金额信号被刻意做得很强；真实欺诈也可能金额正常、时间/频率异常。",
    "因此这里只能说明**本数据集、本参数、前 15 口径**下金额占主导，仍应保留多特征并做敏感性检查。",
]
(ROOT / "generated" / "feature_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
for ax, top, title in [(axes[0], top3, "三特征：amount + txn_hour + daily_cnt"), (axes[1], top_amount, "单特征：amount")]:
    ax.scatter(df.txn_hour, df.amount, s=10, alpha=0.32, color="#9db8d9", edgecolors="none")
    ax.scatter(top.txn_hour, top.amount, s=105, facecolors="none", edgecolors="#d62728", linewidths=1.7)
    ax.set_yscale("log")
    ax.set_xlabel("交易小时（点）")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.annotate("教学虚构数据", xy=(0.99, 0.02), xycoords="axes fraction", ha="right", fontsize=9, color="gray")
axes[0].set_ylabel("交易金额（元，对数轴）")
fig.suptitle(f"只用 amount：前 15 名单与三特征版重合 {overlap}/15", fontsize=14)
fig.tight_layout()
fig.savefig(ROOT / "outputs" / "06_feature_compare.png", dpi=150)
plt.close(fig)
print(f"名单重合 {overlap}/15；顺序完全一致={same_order}")
print("已保存自主实践 1 的名单、分数、报告和图")
