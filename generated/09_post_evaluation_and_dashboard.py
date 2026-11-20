# -*- coding: utf-8 -*-
"""模型冻结后的独立揭盲评估；这是唯一读取 is_fraud 的脚本，同时构建单文件看板。"""
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# 上游模型产物已经生成后才在这里合并标签；标签不回写任何建模脚本或特征文件。
scores = pd.read_csv(ROOT / "generated" / "lof_scores.csv")
labels = pd.read_csv(ROOT / "data" / "transactions.csv", usecols=["txn_id", "is_fraud"])
assert len(scores) == len(labels) == 1060
assert scores.txn_id.is_unique and labels.txn_id.is_unique
assert set(scores.txn_id) == set(labels.txn_id)
assert "is_fraud" not in scores.columns
df = scores.merge(labels, on="txn_id", how="left", validate="one_to_one")
assert df.is_fraud.notna().all()
fraud_total = int(df.is_fraud.sum())
quantiles = [97, 97.25, 97.5, 97.75, 98, 98.25, 98.5, 98.75, 99, 99.25, 99.5]
rows = []
for q in quantiles:
    threshold = float(np.percentile(df.lof_score, q))
    flagged = df.lof_score >= threshold
    hits = int((flagged & (df.is_fraud == 1)).sum())
    count = int(flagged.sum())
    rows.append({
        "q": q, "thr": threshold, "n": count, "hit": hits,
        "miss": fraud_total - hits, "fp": count - hits,
        "precision": hits / count if count else 0, "recall": hits / fraud_total,
        "amount": float(df.loc[flagged, "amount"].sum()),
    })
evaluation = pd.DataFrame(rows)
evaluation.to_csv(ROOT / "generated" / "post_model_evaluation.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
base = evaluation.loc[evaluation.q == 98.5].iloc[0]
lines = [
    "# 模型完成后的独立揭盲评估", "",
    "姓名：龙羲玥｜学号：20251030226｜专业：金融计算机专业", "",
    "> 隔离说明：01—08 建模与分析脚本均不读取 is_fraud。本脚本仅在 LOF 分数、阈值和自主研究冻结后合并标签，用于课程要求的检中数/漏报数展示；标签未参与特征、选参或阈值选择。", "",
    f"- 暗桩共 {fraud_total} 笔（{fraud_total/len(df):.2%}）。",
    f"- 98.5 分位：{int(base.n)} 笔预警，检中 {int(base.hit)}，漏报 {int(base.miss)}，误报 {int(base.fp)}；精确率 {base.precision:.2%}，召回率 {base.recall:.2%}。",
    "- 唯一误报为 T0133（16 点/柜台/超市/2926.63 元），与建模阶段仅凭行为识别出的‘不同笔’一致。",
    "- 这些指标只描述本教学虚构数据，不能外推真实业务表现；看板滑杆也不会重跑模型。", "",
    "## 全阈值结果", "",
    evaluation[["q", "thr", "n", "hit", "miss", "fp", "precision", "recall"]].round({"thr": 4, "precision": 4, "recall": 4}).to_markdown(index=False),
]
(ROOT / "generated" / "post_model_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

points = [{
    "id": r.txn_id, "h": int(r.txn_hour), "a": float(r.amount), "l": float(r.lof_score),
    "f": bool(r.is_fraud), "ch": r.channel, "m": r.merchant_type,
} for r in df.itertuples(index=False)]
sweep = [{"q": float(r.q), "thr": float(r.thr), "n": int(r.n), "hit": int(r.hit), "miss": int(r.miss), "fp": int(r.fp), "amount": float(r.amount)} for r in evaluation.itertuples(index=False)]
meta = {"title": "零售银行反欺诈预警台", "student": "龙羲玥", "studentId": "20251030226", "major": "金融计算机专业", "fraudTotal": fraud_total, "defaultQ": 98.5, "dataNote": "教学虚构数据"}
payload = {"meta": meta, "points": points, "sweep": sweep}
(ROOT / "generated" / "dashboard_data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

html_path = ROOT / "webapp" / "index.html"
html = html_path.read_text(encoding="utf-8")
for name, value in [("PTS", points), ("SWEEP", sweep), ("META", meta)]:
    html, replacements = re.subn(rf"const {name} = .*?;", f"const {name} = {json.dumps(value, ensure_ascii=False, separators=(',', ':'))};", html, count=1)
    assert replacements == 1, f"未找到看板数据占位符：{name}"
html_path.write_text(html, encoding="utf-8")
print(evaluation[["q", "thr", "n", "hit", "miss", "fp"]].round({"thr": 4}).to_string(index=False))
print("已保存独立揭盲评估、dashboard_data.json，并更新零依赖 webapp/index.html")
