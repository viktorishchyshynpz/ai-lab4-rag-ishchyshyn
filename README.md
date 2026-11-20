# Лабораторна робота №4: Побудова рішення пошуково-доповненої генерації (RAG) на базі LLM

- **Студент:** Іщишин Віктор Богданович, група ПЗ-47
- **Дисципліна:** Технології ШІ в інженерії даних
- **Варіант:** 2 (`SEED = 2`)
- **Документ:** OpenStax *Principles of Data Science* — [https://openstax.org/details/books/principles-data-science](https://openstax.org/details/books/principles-data-science)
- **Діапазон сторінок PDF:** сторінки 1–200 (з 569 сторінок оригінального файлу, відповідно до п. 3.1 методичних вказівок)

## Параметри експерименту (Варіант 2)

- **Модель ембедингів:** `sentence-transformers/all-mpnet-base-v2` (`d = 768`, `max_seq_length = 384`)
- **Конфігурації чанкінгу:** `A = (192, 24)`, `B = (384, 48)` (розмір і перекриття в токенах)
- **Локальна LLM:** `qwen2.5:3b` (Ollama, `temperature = 0`, `seed = 2`)
- **Додатковий експеримент (Е2):** переранжування top-20 кандидатів моделлю `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Інструкція запуску середовища

1. Створити та активувати віртуальне середовище Python:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Встановити залежності:
   ```powershell
   pip install -r requirements.txt
   ```
3. Завантажити локальну мовну модель в Ollama:
   ```powershell
   ollama pull qwen2.5:3b
   ```