# src/index.py
# Крок 4. Побудова ембедингів і векторного індексу InMemoryVectorStore

import json
import os
import time
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMB_MODEL, CHUNKS, QUERY_PREFIX, PASSAGE_PREFIX


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def load_chunks(name):
    with open(f"data/chunks_{name}.json", encoding="utf-8") as f:
        return json.load(f)


def load_store(name, emb):
    return InMemoryVectorStore.load(f"index/{name}.json", emb)


def query_text(q):
    return f"{QUERY_PREFIX}{q}" if QUERY_PREFIX else q


if __name__ == "__main__":
    os.makedirs("index", exist_ok=True)
    emb = get_embeddings()
    for name in CHUNKS:
        docs = [
            Document(
                page_content=f"{PASSAGE_PREFIX}{c['text']}" if PASSAGE_PREFIX else c["text"],
                metadata={"chunk_id": c["chunk_id"], "page": c["page"]},
            )
            for c in load_chunks(name)
        ]
        t0 = time.perf_counter()
        store = InMemoryVectorStore.from_documents(docs, emb)
        store.dump(f"index/{name}.json")
        dt = time.perf_counter() - t0
        print(f"Конфігурація {name}: {len(docs)} чанків, індексація {dt:.1f} с")

    # Пробні запити для перевірки коректності пошуку
    print("\n--- Пробні запити (конфігурація B) ---")
    store_b = load_store("B", emb)
    sample_queries = [
        "What is the interquartile range (IQR)?",
        "What are the critical values for 95% confidence interval?",
        "What is the difference between DataFrame and Series in Pandas?",
    ]
    for q in sample_queries:
        print(f"\nЗапит: {q}")
        for doc, score in store_b.similarity_search_with_score(query_text(q), k=3):
            preview = doc.page_content[:95].replace("\n", " ")
            print(f"  score={score:.3f} | стор. {doc.metadata['page']:>3} | {preview}...")
