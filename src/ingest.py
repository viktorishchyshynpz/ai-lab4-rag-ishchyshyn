# src/ingest.py
# Крок 2. Вилучення тексту з PDF і статистика документа в токенах моделі ембедингів

import json
import os
import re
import matplotlib.pyplot as plt
from pypdf import PdfReader
from transformers import AutoTokenizer
from config import DOC_PATH, EMB_MODEL, MAX_PAGES


def load_pages(path, max_pages=MAX_PAGES):
    reader = PdfReader(path)
    total_pdf_pages = len(reader.pages)
    limit = min(total_pdf_pages, max_pages) if max_pages else total_pdf_pages
    pages = []
    for i in range(1, limit + 1):
        raw = reader.pages[i - 1].extract_text() or ""
        text = re.sub(r"\s+", " ", raw).strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages, total_pdf_pages


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    pages, total_pdf_pages = load_pages(DOC_PATH, MAX_PAGES)
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    for p in pages:
        p["tokens"] = len(tok.encode(p["text"], add_special_tokens=False))

    with open("data/pages.json", "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    toks = [p["tokens"] for p in pages]
    print(f"Усього сторінок у PDF: {total_pdf_pages}, взято зріз: 1-{MAX_PAGES}")
    print(f"Непорожніх сторінок: {len(pages)}, всього токенів: {sum(toks)}")
    print(f"Токенів на сторінку: min={min(toks)}, median={sorted(toks)[len(toks)//2]}, "
          f"mean={sum(toks)/len(toks):.1f}, max={max(toks)}")

    # Діагностика вилучення формул і таблиць (особливість варіанта 2)
    sample_pages = {p["page"]: p["text"] for p in pages}
    for pno in (122, 127, 160):
        if pno in sample_pages:
            print(f"\n[Перевірка стор. {pno}] {sample_pages[pno][:260]}...")

    plt.figure(figsize=(10, 4))
    plt.bar([p["page"] for p in pages], toks, color="#2563eb", width=0.8)
    plt.axhline(sum(toks) / len(toks), color="#dc2626", linestyle="--",
                label=f"Середнє: {sum(toks)/len(toks):.0f} ток.")
    plt.xlabel("Номер сторінки PDF")
    plt.ylabel("Кількість токенів")
    plt.title("Розподіл кількості токенів по сторінках (Principles of Data Science, стор. 1-200)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/01_tokens_per_page.png", dpi=150)
    plt.close()
    print("Графік збережено: reports/figures/01_tokens_per_page.png")
    