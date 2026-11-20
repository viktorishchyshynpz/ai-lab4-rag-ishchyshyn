# src/eval_retrieval.py
# Крок 6. Оцінювання пошуку (dense, BM25, гібрид RRF) для конфігурацій A і B

import csv
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from rank_bm25 import BM25Okapi
from index import get_embeddings, load_chunks, load_store, query_text
from config import CHUNKS

K_MAX = 10


def load_questions(path="eval/questions.csv"):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["gold"] = {int(p) for p in r["pages"].split(";") if p.strip()}
    return rows


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def bm25_ranking(bm25, ids, q, k=K_MAX):
    scores = bm25.get_scores(tokenize(q))
    return [ids[i] for i in np.argsort(scores)[::-1][:k]]


def rrf(*rankings, k=60):
    score = {}
    for ranking in rankings:
        for r, cid in enumerate(ranking, start=1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + r)
    return sorted(score, key=score.get, reverse=True)


def hit_at_k(retrieved_pages, gold, k):
    return int(bool(set(retrieved_pages[:k]) & gold))


def reciprocal_rank(retrieved_pages, gold):
    for r, p in enumerate(retrieved_pages, start=1):
        if p in gold:
            return 1.0 / r
    return 0.0


if __name__ == "__main__":
    os.makedirs("eval", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    questions = load_questions()
    answerable = [q for q in questions if q["gold"]]
    unanswerable = [q for q in questions if not q["gold"]]
    emb = get_embeddings()

    rankings = {}
    top1_by_cfg = {}

    for name in CHUNKS:
        chunks = load_chunks(name)
        ids = [c["chunk_id"] for c in chunks]
        page_of = {c["chunk_id"]: c["page"] for c in chunks}
        bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        store = load_store(name, emb)

        top1 = {"answerable": [], "unanswerable": []}
        for q in questions:
            dense_hits = store.similarity_search_with_score(query_text(q["question"]), k=K_MAX)
            dense_ids = [d.metadata["chunk_id"] for d, _ in dense_hits]
            bm25_ids = bm25_ranking(bm25, ids, q["question"], k=K_MAX)
            hyb_ids = rrf(dense_ids, bm25_ids)[:K_MAX]

            bucket = "answerable" if q["gold"] else "unanswerable"
            top1[bucket].append(dense_hits[0][1])

            for method, ranking in (("dense", dense_ids), ("bm25", bm25_ids), ("hybrid", hyb_ids)):
                rankings[(f"{name}-{method}", q["qid"])] = [page_of[c] for c in ranking]
        top1_by_cfg[name] = top1

    systems = sorted({s for s, _ in rankings})
    ks = list(range(1, K_MAX + 1))

    plt.figure(figsize=(9, 5))
    with open("eval/retrieval_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR"])
        print(f"{'System':<12} {'Hit@1':>6} {'Hit@3':>6} {'Hit@5':>6} {'Hit@10':>7} {'MRR':>6}")
        for s in systems:
            curve = [
                np.mean([hit_at_k(rankings[(s, q["qid"])], q["gold"], k) for q in answerable])
                for k in ks
            ]
            mrr = np.mean([reciprocal_rank(rankings[(s, q["qid"])], q["gold"]) for q in answerable])
            values = (curve[0], curve[2], curve[4], curve[9], mrr)
            w.writerow([s] + [f"{v:.2f}" for v in values])
            print(f"{s:<12} {values[0]:6.2f} {values[1]:6.2f} {values[2]:6.2f} {values[3]:7.2f} {values[4]:6.2f}")
            plt.plot(ks, curve, marker="o", label=s)

    plt.xlabel("k")
    plt.ylabel("Hit@k")
    plt.title("Криві Hit@k для шести систем пошуку (Варіант 2)")
    plt.ylim(0, 1.05)
    plt.xticks(ks)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/03_hit_at_k.png", dpi=150)
    plt.close()

    # Розподіл top-1 схожості для конфігурації B (або кращої)
    best_cfg = "B"
    top1 = top1_by_cfg[best_cfg]
    print(f"\n--- Статистика top-1 косинусної схожості (конфігурація {best_cfg}) ---")
    print(f"Є відповідь (n={len(top1['answerable'])}): "
          f"min={min(top1['answerable']):.3f}, median={np.median(top1['answerable']):.3f}, "
          f"max={max(top1['answerable']):.3f}")
    print(f"Немає відповіді (n={len(top1['unanswerable'])}): "
          f"min={min(top1['unanswerable']):.3f}, median={np.median(top1['unanswerable']):.3f}, "
          f"max={max(top1['unanswerable']):.3f}")

    # Детальна діагностика по запитаннях для відповідей на питання кроку 6
    print("\n--- Позапитальна діагностика (Hit@5 та ранг першого влучання) ---")
    for q in answerable:
        hits_str = " | ".join(
            f"{s}:{reciprocal_rank(rankings[(s, q['qid'])], q['gold']):.2f}"
            for s in ("B-dense", "B-bm25", "B-hybrid")
        )
        print(f"{q['qid']} [{q['type']:<10}] gold={sorted(q['gold'])} -> {hits_str}")

    plt.figure(figsize=(8, 4))
    plt.hist(top1["answerable"], bins=12, alpha=0.65, label="є відповідь", color="#2563eb")
    plt.hist(top1["unanswerable"], bins=12, alpha=0.65, label="немає відповіді", color="#ea580c")
    plt.xlabel("Косинусна схожість top-1 чанка")
    plt.ylabel("Кількість запитань")
    plt.title(f"Розподіл схожості top-1 (конфігурація {best_cfg}-dense)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/04_top1_scores.png", dpi=150)
    plt.close()
    print("\nЗбережено: eval/retrieval_metrics.csv, 03_hit_at_k.png, 04_top1_scores.png")
