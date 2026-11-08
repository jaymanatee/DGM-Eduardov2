"""Training curves of the two GRPO runs on one figure: reports/phase1_grpo_curves.png.

Both runs go on the same x axis, problems seen (steps x problems per step), so a run with one
problem per step and a run with three can be compared. Thick line: moving average over
``--window`` problems; thin line: every step.

Run (needs both adapters in rlm/weights, see docs/fase1_reproducir.md)::

    uv run python -m rlm.plot_grpo_curves
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# (label, adapter folder with log_history.json, problems per step, colour)
RUNS = [
    ("GRPO 1: 1 problema/paso, 400 pasos", "rlm/weights/final_rlm_lora", 1, "#2a78d6"),
    ("GRPO 2: 3 problemas/paso, 270 pasos", "rlm/weights/grpo_pps3", 3, "#eb6834"),
]
PANELS = [
    ("rewards/origin_accuracy_reward/mean", "Recompensa de exactitud"),
    ("rewards/format_reward/mean", "Recompensa de formato"),
    ("completions/clipped_ratio", "Fracción de respuestas cortadas"),
    ("completions/mean_length", "Longitud media (tokens)"),
]
INK, MUTED, GRID, PAPER = "#1f1f1e", "#6b6a63", "#e6e5e0", "#fcfcfb"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--window", type=int, default=60, help="moving average, in problems")
    parser.add_argument("--out", default="reports/phase1_grpo_curves.png")
    args = parser.parse_args()

    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, facecolor=PAPER)
    for ax, (key, title) in zip(axes.flat, PANELS, strict=True):
        ax.set_facecolor(PAPER)
        for label, path, per_step, color in RUNS:
            history = json.loads(Path(path, "log_history.json").read_text(encoding="utf-8"))
            rows = [r for r in history if key in r]
            x = np.array([r["step"] * per_step for r in rows])
            y = np.array([r[key] for r in rows])
            w = max(1, args.window // per_step)
            smooth = np.convolve(y, np.ones(w) / w, mode="valid")
            ax.plot(x, y, color=color, lw=0.8, alpha=0.18)
            ax.plot(x[w - 1 :], smooth, color=color, lw=2, label=label)
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Problemas vistos (pasos × problemas por paso)")
    axes[0, 0].legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle(
        f"Curvas de entrenamiento GRPO (línea gruesa: media móvil de {args.window} problemas; "
        "fina: cada paso)",
        x=0.01,
        ha="left",
        fontsize=12,
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    print(f"curves -> {out}")


if __name__ == "__main__":
    main()
