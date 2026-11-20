# src/git_timeline.py
# Крок 10. Історія комітів і часова шкала виконання роботи

import os
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt

os.makedirs("reports/figures", exist_ok=True)
log = subprocess.run(
    ["git", "log", "--reverse", "--pretty=format:%h|%at|%s"],
    capture_output=True, text=True, check=True,
).stdout

rows = [line.split("|", 2) for line in log.splitlines() if line.strip()]
t0 = int(rows[0][1])
minutes = [(int(t) - t0) / 60 for _, t, _ in rows]

for (h, t, msg), m in zip(rows, minutes):
    when = datetime.fromtimestamp(int(t))
    print(f"{h}  {when:%Y-%m-%d %H:%M}  +{m:6.0f} хв  {msg}")
print(f"Загальна тривалість: {minutes[-1] / 60:.1f} год, комітів: {len(rows)}")

plt.figure(figsize=(8, 4))
plt.step(minutes, range(1, len(rows) + 1), where="post", marker="o", color="#2563eb")
for (h, _, msg), m, y in zip(rows, minutes, range(1, len(rows) + 1)):
    plt.annotate(
        msg.split(":")[0], (m, y),
        textcoords="offset points", xytext=(5, -10), fontsize=7,
    )
plt.xlabel("Хвилин від першого коміту")
plt.ylabel("Номер коміту")
plt.title("Часова шкала комітів (ai-lab4-rag-ishchyshyn)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("reports/figures/07_git_timeline.png", dpi=150)
plt.close()
