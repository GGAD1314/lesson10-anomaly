# -*- coding: utf-8 -*-
"""数据初检与全景可视化；只读取七个非标注字段。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transactions.csv"
OUT = ROOT / "outputs" / "01_raw_scatter.png"
SAFE_COLUMNS = ["txn_id", "customer_id", "txn_date", "txn_hour", "amount", "channel", "merchant_type"]

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Songti SC", "STHeiti", "PingFang SC"]
matplotlib.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(DATA, usecols=SAFE_COLUMNS)
print("原始数据：1060 行 × 8 列；分析视图（未读取 is_fraud）：", df.shape)
print("分析视图缺失值总数：", int(df.isna().sum().sum()))
print(df[["txn_hour", "amount"]].describe().round(2).to_string())

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(df["txn_hour"], df["amount"], s=14, alpha=0.45, color="#3b6fb6", edgecolors="none")
ax.set_yscale("log")
ax.set_xticks(range(1, 22, 2))
ax.set_xlabel("交易小时（点）")
ax.set_ylabel("交易金额（元，对数轴）")
ax.set_title("1060 笔交易全景：交易小时—金额")
ax.grid(alpha=0.3)
ax.annotate("教学虚构数据", xy=(0.99, 0.01), xycoords="axes fraction", ha="right", fontsize=9, color="gray")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
plt.close(fig)
print(f"已保存 {OUT.relative_to(ROOT)}")
