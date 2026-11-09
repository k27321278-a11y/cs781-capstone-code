"""Plot archived Phase 1 LegalSpan training loss against realized target tokens."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).resolve().parent
with (here / "figure_data" / "phase1_loss_points.csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
x = np.array([int(r["target_tokens"]) / 1e6 for r in rows])
y = np.array([float(r["loss"]) for r in rows])
smooth = np.array([np.median(y[max(0, i - 12):i + 13]) for i in range(len(y))])

fig, ax = plt.subplots(figsize=(5.8, 2.5))
ax.plot(x, y, color="#9dbad5", linewidth=0.5, alpha=0.45, label="logged loss")
ax.plot(x, smooth, color="#14568b", linewidth=1.6, label="25-record rolling median")
ax.set(xlabel="Target tokens processed (millions)", ylabel="Training NLL")
ax.grid(alpha=0.22)
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(here / "figure_phase1_training_loss.pdf", bbox_inches="tight")
