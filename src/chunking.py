# src/chunking.py
# Крок 3. Чанкінг у конфігураціях A (192, 24) і B (384, 48)

import json
import os
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import EMB_MODEL, CHUNKS


def make_chunks(pages, size, overlap, tok):
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tok,
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = [
        Document(page_content=p["text"], metadata={"page": p["page"]})
        for p in pages
    ]
    chunks = splitter.split_documents(docs)
    return [
        {"chunk_id": i, "page": c.metadata["page"], "text": c.page_content}
        for i, c in enumerate(chunks)
    ]


if __name__ == "__main__":
    os.makedirs("reports/figures", exist_ok=True)
    with open("data/pages.json", encoding="utf-8") as f:
        pages = json.load(f)

    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    max_len = tok.model_max_length
    print(f"Токенізатор: {EMB_MODEL}, model_max_length={max_len}")

    lengths = {}
    for name, (size, overlap) in CHUNKS.items():
        chunks = make_chunks(pages, size, overlap, tok)
        with open(f"data/chunks_{name}.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        lens = [len(tok.encode(c["text"], add_special_tokens=False)) for c in chunks]
        lengths[name] = lens
        over_limit = sum(1 for x in lens if x > 384)
        print(f"Конфігурація {name} ({size}, {overlap}): {len(chunks)} чанків, "
              f"min={min(lens)}, median={sorted(lens)[len(lens)//2]}, "
              f"mean={sum(lens)/len(lens):.1f}, max={max(lens)}, >384={over_limit}")

    plt.figure(figsize=(8, 4))
    for name, lens in lengths.items():
        size, overlap = CHUNKS[name]
        plt.hist(lens, bins=25, alpha=0.6, label=f"{name} ({size}, {overlap}) — {len(lens)} чанків")
    plt.axvline(384, color="#dc2626", linestyle="--", label="max_seq_length = 384")
    plt.xlabel("Довжина чанка, токенів")
    plt.ylabel("Кількість чанків")
    plt.title("Розподіл довжин чанків у конфігураціях A і B")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/02_chunk_lengths.png", dpi=150)
    plt.close()
    print("Графік збережено: reports/figures/02_chunk_lengths.png")
