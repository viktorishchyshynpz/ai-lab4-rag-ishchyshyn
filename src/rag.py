# src/rag.py
# Крок 7. RAG-ланцюжок LCEL та генерація відповідей у режимах rag і llm

import csv
import os
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from index import get_embeddings, load_store, query_text
from eval_retrieval import load_questions
from config import BEST_CONFIG, LLM_MODEL, TOP_K, THRESHOLD, SEED

REFUSAL = "НЕМАЄ У ДОКУМЕНТІ"
SYSTEM = (
    "Ти помічник, який відповідає українською мовою ЛИШЕ на основі наданого "
    "контексту (контекст може бути англійською мовою). Після кожного твердження "
    "вкажи джерело у форматі [стор. N]. "
    f"Якщо в контексті немає відповіді, напиши дослівно: {REFUSAL}."
)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Контекст:\n{context}\n\nПитання: {question}"),
])
plain_prompt = ChatPromptTemplate.from_messages([
    ("human", "Відповідай українською мовою стисло й точно.\n\nПитання: {question}"),
])

llm = ChatOllama(model=LLM_MODEL, temperature=0, seed=SEED)
rag_chain = rag_prompt | llm | StrOutputParser()      # LCEL: prompt -> LLM -> текст
plain_chain = plain_prompt | llm | StrOutputParser()  # базова лінія без RAG


def format_docs(docs):
    return "\n\n".join(f"[стор. {d.metadata['page']}] {d.page_content}" for d in docs)


def ask(question, store, k=TOP_K, threshold=THRESHOLD):
    t0 = time.perf_counter()
    hits = store.similarity_search_with_score(query_text(question), k=k)
    t1 = time.perf_counter()

    if not hits or hits[0][1] < threshold:
        return {
            "answer": REFUSAL,
            "pages": "",
            "top1_score": round(hits[0][1], 4) if hits else 0.0,
            "t_retr": round(t1 - t0, 4),
            "t_gen": 0.0,
        }

    docs = [d for d, _ in hits]
    context = format_docs(docs)
    t2 = time.perf_counter()
    answer = rag_chain.invoke({"context": context, "question": question}).strip()
    t3 = time.perf_counter()

    pages_str = ";".join(str(d.metadata["page"]) for d in docs)
    return {
        "answer": answer,
        "pages": pages_str,
        "top1_score": round(hits[0][1], 4),
        "t_retr": round(t1 - t0, 4),
        "t_gen": round(t3 - t2, 4),
    }


def ask_plain(question):
    t0 = time.perf_counter()
    answer = plain_chain.invoke({"question": question}).strip()
    t1 = time.perf_counter()
    return {
        "answer": answer,
        "pages": "",
        "top1_score": "",
        "t_retr": 0.0,
        "t_gen": round(t1 - t0, 4),
    }


if __name__ == "__main__":
    os.makedirs("eval", exist_ok=True)
    questions = load_questions()
    emb = get_embeddings()
    store = load_store(BEST_CONFIG, emb)

    rows = []
    for q in questions:
        print(f"Обробка {q['qid']} ({q['type']})...")
        res_rag = ask(q["question"], store, k=TOP_K, threshold=THRESHOLD)
        rows.append({
            "qid": q["qid"],
            "mode": "rag",
            "answer": res_rag["answer"],
            "pages_retrieved": res_rag["pages"],
            "top1_score": res_rag["top1_score"],
            "t_retr": res_rag["t_retr"],
            "t_gen": res_rag["t_gen"],
            "label": "",
            "citation_ok": "",
        })
        res_llm = ask_plain(q["question"])
        rows.append({
            "qid": q["qid"],
            "mode": "llm",
            "answer": res_llm["answer"],
            "pages_retrieved": "",
            "top1_score": "",
            "t_retr": res_llm["t_retr"],
            "t_gen": res_llm["t_gen"],
            "label": "",
            "citation_ok": "",
        })

    fieldnames = [
        "qid", "mode", "answer", "pages_retrieved", "top1_score",
        "t_retr", "t_gen", "label", "citation_ok",
    ]
    with open("eval/answers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Готово! Збережено {len(rows)} відповідей у eval/answers.csv")
    print("Далі скопіюйте eval/answers.csv у eval/answers_labeled.csv і заповніть вручну колонки label та citation_ok.")
