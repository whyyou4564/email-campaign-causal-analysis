"""保存标准 scikit-uplift Qini 曲线，明确区别于总体 IPW 增益曲线。"""
import os
from pathlib import Path

_cache = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
from sklift.metrics import qini_curve

from .config import OUT, FIG

def standard_qini():
    predictions = pd.read_csv(OUT / "test_predictions.csv")
    colors = {
        "Response prediction": "#64748b", "S-Learner": "#2563eb",
        "T-Learner": "#16a34a", "X-Learner": "#9333ea",
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    rows = []
    for model, color in colors.items():
        sample_counts, gains = qini_curve(
            predictions.conversion.to_numpy(), predictions[model].to_numpy(),
            predictions.treatment.to_numpy(),
        )
        fractions = sample_counts / len(predictions)
        ax.plot(fractions, gains, color=color, label=model, linewidth=1.8)
        for fraction, gain in zip(fractions, gains):
            rows.append({"model": model, "fraction": fraction, "qini_gain_treated_scale": gain})
    ax.plot([0, 1], [0, gains[-1]], "--", color="#94a3b8", label="Random expected")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Share of held-out customers targeted")
    ax.set_ylabel("Qini gain: treated-scale excess purchase events")
    ax.set_title("Standard Qini curves | scikit-uplift definition", fontweight="bold")
    ax.text(0, -.16, "This scale differs from population incremental events in the IPW gain curves.",
            transform=ax.transAxes, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "09-qini-standard.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "qini_curves_sklift.csv", index=False)
