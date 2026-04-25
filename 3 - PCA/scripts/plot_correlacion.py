import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_circulo_correlaciones(
    col_corr: pd.DataFrame, explained: np.ndarray, title: str = "Circulo de correlaciones"
) -> None:
    """Grafica circulo de correlaciones para PC1 y PC2."""
    fig, ax = plt.subplots(figsize=(7, 7))
    circle = plt.Circle((0, 0), 1, color="steelblue", fill=False, linestyle="--", linewidth=1.5)
    ax.add_artist(circle)

    for var in col_corr.index:
        x_val = col_corr.loc[var, "PC1"]
        y_val = col_corr.loc[var, "PC2"]
        ax.arrow(
            0,
            0,
            x_val,
            y_val,
            color="darkred",
            alpha=0.75,
            head_width=0.03,
            head_length=0.04,
            length_includes_head=True,
        )
        ax.text(x_val * 1.1, y_val * 1.1, var, color="black", ha="center", va="center")

    ax.axhline(0, color="gray", linewidth=1)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel(f"PC1 ({explained[0]:.2f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1]:.2f}% var)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()

