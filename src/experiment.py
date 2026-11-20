# src/experiment.py
# Додатковий експеримент Е2 (Варіант 2):
# Переранжування top-20 кандидатів моделлю cross-encoder/ms-marco-MiniLM-L-6-v2

import csv
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import CrossEncoder
from index import get_embeddings, load_store, query_text
from eval_retrieval import load_questions, hit_at_k, reciprocal_rank
from config import BEST_CONFIG, RERANK_MODEL

FETCH_K = 20
EVAL_KS = [1, 3, 5]


if __name__ == "__main__":
    os.makedirs("eval", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    questions = load_questions()
    answerable = [q for q in questions if q["gold"]]

    emb = get_embeddings()
    store = load_store(BEST_CONFIG, emb)
    reranker = CrossEncoder(RERANK_MODEL)

    base_rankings, rerank_rankings = {}, {}
    base_times, rerank_times = [], []

    for q in answerable:
        # 1. Базовий dense-пошук
        t0 = time.perf_counter()
        hits = store.similarity_search_with_score(query_text(q["question"]), k=FETCH_K)
        t1 = time.perf_counter()
        base_times.append(t1 - t0)
        base_pages = [d.metadata["page"] for d, _ in hits]
        base_rankings[q["qid"]] = base_pages

        # 2. Переранжування top-20 через CrossEncoder
        t2 = time.perf_counter()
        pairs = [(q["question"], d.page_content) for d, _ in hits]
        ce_scores = reranker.predict(pairs)
        order = np.argsort(ce_scores)[::-1]
        rerank_pages = [hits[i][0].metadata["page"] for i in order]
        t3 = time.perf_counter()
        rerank_times.append((t1 - t0) + (t3 - t2))
        rerank_rankings[q["qid"]] = rerank_pages

    results = []
    for label, rk, times in (
        (f"{BEST_CONFIG}-dense (базовий)", base_rankings, base_times),
        (f"{BEST_CONFIG}-dense + E2 CrossEncoder", rerank_rankings, rerank_times),
    ):
        h1 = np.mean([hit_at_k(rk[q["qid"]], q["gold"], 1) for q in answerable])
        h3 = np.mean([hit_at_k(rk[q["qid"]], q["gold"], 3) for q in answerable])
        h5 = np.mean([hit_at_k(rk[q["qid"]], q["gold"], 5) for q in answerable])
        mrr = np.mean([reciprocal_rank(rk[q["qid"]], q["gold"]) for q in answerable])
        avg_ms = np.mean(times) * 1000
        results.append((label, h1, h3, h5, mrr, avg_ms))

    with open("eval/experiment_e2_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "Hit@1", "Hit@3", "Hit@5", "MRR", "avg_latency_ms"])
        print(f"{'System':<32} {'Hit@1':>6} {'Hit@3':>6} {'Hit@5':>6} {'MRR':>6} {'Latency(ms)':>12}")
        for row in results:
            w.writerow([row[0], f"{row[1]:.2f}", f"{row[2]:.2f}", f"{row[3]:.2f}", f"{row[4]:.2f}", f"{row[5]:.1f}"])
            print(f"{row[0]:<32} {row[1]:6.2f} {row[2]:6.2f} {row[3]:6.2f} {row[4]:6.2f} {row[5]:12.1f}")

    # Побудова порівняльного графіка
    metrics = ["Hit@1", "Hit@3", "Hit@5", "MRR"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 1.3]})
    ax1.bar(x - width / 2, results[0][1:5], width, label=results[0][0], color="#64748b")
    ax1.bar(x + width / 2, results[1][1:5], width, label=results[1][0], color="#2563eb")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Значення метрики")
    ax1.set_title("Експеримент Е2: якість пошуку")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.bar(["Dense", "+Rerank"], [results[0][5], results[1][5]], color=["#64748b", "#2563eb"])
    ax2.set_ylabel("Середній час пошуку, мс")
    ax2.set_title("Час пошуку")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("reports/figures/08_experiment_e2_rerank.png", dpi=150)
    plt.close()
    print("\nЗбережено: eval/experiment_e2_metrics.csv, reports/figures/08_experiment_e2_rerank.png")
