from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "caa_tuning_20260916" / "caa_parameter_sweep.csv"
OUT = DATA.with_name("caa_parameter_sweep.png")


def main() -> None:
    frame = pd.read_csv(DATA)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=False)
    colors = {"Qwen3.6-27B": "#4C78A8", "Gemma-4-31B-IT": "#E45756"}

    for ax, model in zip(axes, ("Qwen3.6-27B", "Gemma-4-31B-IT")):
        part = frame[frame["model"] == model].copy()
        part["label"] = part.apply(
            lambda row: f"L{row.layer}, a={row.alpha:g}" + (" (retry)" if row.variant == "recovered" else ""),
            axis=1,
        )
        y = np.arange(len(part))
        height = 0.34
        ax.barh(y - height / 2, part["accuracy_percent"], height=height, color=colors[model], alpha=0.9, label="Accuracy")
        ax.barh(y + height / 2, part["generation_failure_percent"], height=height, color="#B9BDC5", label="Generation failure rate")
        ax.set_yticks(y, part["label"])
        ax.invert_yaxis()
        ax.set_xlim(0, 60)
        ax.set_xlabel("Percent of 500 questions")
        ax.set_title(model)
        ax.grid(axis="x", alpha=0.2)
        for pos, value in zip(y, part["accuracy_percent"]):
            ax.text(value + 0.7, pos - height / 2, f"{value:.1f}%", va="center", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False)
    fig.suptitle("HLE with Tools: Outcome-CAA parameter sweep", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.86))
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
