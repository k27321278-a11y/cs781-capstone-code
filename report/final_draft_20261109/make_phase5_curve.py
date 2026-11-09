"""Render an evidence-based Phase 5 calibration curve from preserved run logs."""
import json
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
SOURCES = {
    "Sparse-guided": "phase5_b1_sparse_guided_training_history.json",
    "Position queries": "phase5_b1_position_queries_training_history.json",
    "Statute GRU": "phase5_b1_statute_gru_training_history.json",
}

fig, ax = plt.subplots(figsize=(5.8, 2.6))
for label, name in SOURCES.items():
    rows = json.loads((HERE / "figure_data" / name).read_text(encoding="utf-8"))
    ax.plot(
        [row["epoch"] for row in rows],
        [row["calibration"]["macro_f1"] for row in rows],
        marker="o",
        linewidth=1.5,
        label=label,
    )
ax.set(xlabel="Epoch", ylabel="Train-calibration macro-F1", xticks=[1, 2, 3, 4])
ax.set_ylim(0.17, 0.26)
ax.grid(alpha=0.22)
ax.legend(frameon=False, ncol=3, fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(HERE / "figure_phase5_calibration.pdf", bbox_inches="tight")
