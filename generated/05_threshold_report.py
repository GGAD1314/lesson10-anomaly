# -*- coding: utf-8 -*-
"""Step2：98.5 分位阈值、旗舰名单、阈值天平图和三段式风控报告。"""
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
df["lof_score"] = -LocalOutlierFactor(n_neighbors=20).fit(X).negative_outlier_factor_

thr = np.percentile(df.lof_score, 98.5)
flag = df[df.lof_score >= thr].copy().sort_values("lof_score", ascending=False)
cols = ["txn_id", "customer_id", "txn_date", "txn_hour", "amount", "channel", "merchant_type", "lof_score"]
flag[cols].to_csv(ROOT / "generated" / "flagship_list.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
odd = flag[(flag.txn_hour > 6) | (flag.channel == "柜台")]
odd_ids = odd.txn_id.tolist()
quantiles = [97, 97.25, 97.5, 97.75, 98, 98.25, 98.5, 98.75, 99, 99.25, 99.5]
sweep = pd.DataFrame({"quantile": quantiles, "threshold": [np.percentile(df.lof_score, q) for q in quantiles]})
sweep["count"] = [(df.lof_score >= t).sum() for t in sweep.threshold]
sweep["amount_sum"] = [df.loc[df.lof_score >= t, "amount"].sum() for t in sweep.threshold]
sweep.to_csv(ROOT / "generated" / "threshold_sweep.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
print(f"旗舰名单：{len(flag)} 笔；阈值：{thr:.6f}；金额合计：{flag.amount.sum():,.2f} 元；不同笔：{odd_ids}")
print(sweep.to_string(index=False, formatters={"threshold": "{:.4f}".format}))

removed = set(flag.txn_id) - set(df.loc[df.lof_score >= np.percentile(df.lof_score, 99.5), "txn_id"])
added = set(df.loc[df.lof_score >= np.percentile(df.lof_score, 97), "txn_id"]) - set(flag.txn_id)
with pd.option_context("display.max_rows", 40):
    print("99.5 相比 98.5 移除：", df[df.txn_id.isin(removed)][["txn_id", "txn_hour", "amount"]].sort_values("txn_id").to_string(index=False))
    print("97 相比 98.5 新增：", df[df.txn_id.isin(added)][["txn_id", "txn_hour", "amount"]].sort_values("txn_id").to_string(index=False))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.2))
positive = df[df.lof_score > 0]
ax1.hist(positive.lof_score, bins=np.logspace(np.log10(positive.lof_score.min()), np.log10(positive.lof_score.max()), 60), color="#3b6fb6", alpha=0.75)
ax1.set_xscale("log")
ax1.axvline(thr, color="#d62728", linestyle="--", linewidth=1.6, label=f"98.5 分位阈值 = {thr:.2f}")
ax1.scatter(flag.lof_score, np.full(len(flag), 2), color="#d62728", s=18, alpha=0.75, label="阈值右侧名单")
ax1.set_xlabel("LOF 分数（对数轴）")
ax1.set_ylabel("交易笔数")
ax1.set_title("阈值天平：调松淹没风险↑，调严漏报风险↑")
ax1.legend()
ax1.grid(alpha=0.3, which="both")
ax1.annotate("教学虚构数据", xy=(0.99, 0.75), xycoords="axes fraction", ha="right", fontsize=9, color="gray")
ax2.scatter(df.txn_hour, df.amount, s=10, alpha=0.3, color="#9db8d9", edgecolors="none")
ax2.scatter(flag.txn_hour, flag.amount, s=110, facecolors="none", edgecolors="#d62728", linewidths=1.8, label=f"旗舰名单 {len(flag)} 笔")
if len(odd):
    ax2.scatter(odd.txn_hour, odd.amount, s=150, marker="s", facecolors="none", edgecolors="#e8963c", linewidths=2)
    for _, r in odd.iterrows():
        ax2.annotate(r.txn_id, (r.txn_hour, r.amount), textcoords="offset points", xytext=(8, 6), fontsize=9, color="#b66a18")
ax2.set_yscale("log")
ax2.set_xlabel("交易小时（点）")
ax2.set_ylabel("交易金额（元，对数轴）")
ax2.set_title(f"旗舰名单在小时—金额平面（橙色方框 = {len(odd)} 笔不同交易）")
ax2.legend()
ax2.grid(alpha=0.3)
ax2.annotate("教学虚构数据", xy=(0.99, 0.02), xycoords="axes fraction", ha="right", fontsize=9, color="gray")
fig.tight_layout()
fig.savefig(ROOT / "outputs" / "05_threshold_balance.png", dpi=150)
plt.close(fig)

odd_desc = "、".join(f"{r.txn_id}（{r.txn_hour} 点/{r.channel}/{r.merchant_type}/{r.amount:,.2f} 元）" for _, r in odd.iterrows()) or "无"
lines = [
    "# 致风控经理：本期异常交易旗舰名单", "",
    "姓名：龙羲玥｜学号：20251030226｜专业：金融计算机专业", "",
    "## ① 我们抓了什么",
    f"共 {len(flag)} 笔，涉案金额 {flag.amount.sum():,.2f} 元；{len(flag)-len(odd)} 笔集中在凌晨 1—4 点、1.66 万—4.37 万元。",
    "主体为网银/APP 大额交易，与白天主体形成孤岛，先列入复核队列而非直接定罪。", "",
    "## ② 我们可能抓错了什么",
    f"不同笔为 {odd_desc}；它是白天柜台消费，只因局部密度较高分被拖入，属于淹没嫌疑。",
    "一刀切会造成正常客户支付失败、投诉和人工复核成本；阈值划线不等于欺诈判决。", "",
    "## ③ 建议动作",
    f"凌晨大额 {len(flag)-len(odd)} 笔先临时限额并人工回访，确认后放行或升级调查；T0133 仅观察、不拦截。",
    "最得罪客户的是‘全拦’；最放过 fraudster 的是‘只看不处置’，所以用分层动作平衡体验与资损。", "",
    "## 阈值验证问答",
    f"- 99.5 分位：{int(sweep.loc[sweep["quantile"] == 99.5, 'count'].iloc[0])} 笔，比 98.5 少 {len(flag)-int(sweep.loc[sweep["quantile"] == 99.5, 'count'].iloc[0])} 笔；移除 {len(removed)} 笔，交易号见下。",
    "  " + "、".join(sorted(removed)) + "。",
    f"- 97 分位：{int(sweep.loc[sweep["quantile"] == 97, 'count'].iloc[0])} 笔，比 98.5 多 {int(sweep.loc[sweep["quantile"] == 97, 'count'].iloc[0])-len(flag)} 笔；新增交易号见下。",
    "  " + "、".join(sorted(added)) + "。", "",
    "---",
    "方法与局限：LOF（k=20，标准化三特征：金额/小时/当日笔数），98.5 分位阈值；未读取 is_fraud。教学虚构数据，不能推断真实欺诈率。",
]
(ROOT / "generated" / "risk_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("已保存 flagship_list.csv、threshold_sweep.csv、risk_report.md 和 05_threshold_balance.png")
