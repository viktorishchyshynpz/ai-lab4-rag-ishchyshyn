# src/eval_answers.py
# Крок 8. Метрики якості відповідей для режимів rag і llm та графіки 05 і 06

import os
import pandas as pd
import matplotlib.pyplot as plt

LABELS = [
    "correct", "partial", "wrong",
    "hallucination", "refusal_ok", "refusal_wrong",
]

os.makedirs("reports/figures", exist_ok=True)
df = pd.read_csv("eval/answers_labeled.csv")
q = pd.read_csv("eval/questions.csv", dtype=str).fillna("")
df = df.merge(q[["qid", "pages"]].rename(columns={"pages": "gold"}), on="qid")
assert df["label"].isin(LABELS).all(), "Розмітьте вручну всі відповіді у eval/answers_labeled.csv"

print("--- Підсумкові метрики відповідей ---")
for mode, g in df.groupby("mode"):
    ans, unans = g[g["gold"] != ""], g[g["gold"] == ""]
    correct = ((ans["label"] == "correct") + 0.5 * (ans["label"] == "partial")).mean()
    halluc = (g["label"] == "hallucination").mean()
    refusal = (unans["label"] == "refusal_ok").mean()
    ref_wrong = int((ans["label"] == "refusal_wrong").sum())
    print(f"{mode}: правильність={correct:.2f}, галюцинації={halluc:.2f}, "
          f"коректні відмови={refusal:.2f}, хибні відмови={ref_wrong}")

rag_cit = df[(df["mode"] == "rag") & df["citation_ok"].notna()]
if len(rag_cit) > 0:
    print(f"Точність посилань на сторінки (rag): {rag_cit['citation_ok'].mean():.2f}")

tab = pd.crosstab(df["mode"], df["label"], normalize="index")
tab = tab.reindex(columns=LABELS, fill_value=0)
tab.plot.barh(stacked=True, figsize=(8, 3.2))
plt.xlabel("Частка відповідей")
plt.title("Розподіл міток відповідей у режимах rag та llm")
plt.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28))
plt.tight_layout()
plt.savefig("reports/figures/05_answer_labels.png", dpi=150)
plt.close()

r = df[df["mode"] == "rag"]
gen_share = r["t_gen"].sum() / (r["t_retr"].sum() + r["t_gen"].sum()) * 100
print(f"Середній час пошуку: {r['t_retr'].mean():.4f} с, генерації: {r['t_gen'].mean():.2f} с "
      f"(частка генерації: {gen_share:.1f}%)")

plt.figure(figsize=(6, 4))
plt.boxplot([r["t_retr"], r["t_gen"]], tick_labels=["пошук", "генерація"])
plt.ylabel("Час, с")
plt.yscale("log")
plt.title("Розподіл часу пошуку та генерації (режим rag)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("reports/figures/06_latency.png", dpi=150)
plt.close()
print("Збережено: reports/figures/05_answer_labels.png, reports/figures/06_latency.png")
