import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_biplot_sobreposicion(
    row_coords: pd.DataFrame, col_corr: pd.DataFrame, classes: pd.Series, explained: np.ndarray
) -> None:
    """Grafico de sobreposicion (biplot) de individuos y variables."""
    fig, ax = plt.subplots(figsize=(9, 7))
    points_df = row_coords.copy()
    points_df["class"] = classes.values

    sns.scatterplot(data=points_df, x="PC1", y="PC2", hue="class", s=55, alpha=0.7, ax=ax)

    scale_x = np.max(np.abs(row_coords["PC1"]))
    scale_y = np.max(np.abs(row_coords["PC2"]))
    scale = min(scale_x, scale_y) * 0.8

    for var in col_corr.index:
        vx = col_corr.loc[var, "PC1"] * scale
        vy = col_corr.loc[var, "PC2"] * scale
        ax.arrow(
            0,
            0,
            vx,
            vy,
            color="darkred",
            alpha=0.8,
            head_width=0.08,
            head_length=0.12,
            length_includes_head=True,
        )
        ax.text(vx * 1.08, vy * 1.08, var, color="darkred", fontsize=9)

    ax.axhline(0, color="gray", linewidth=1)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_title("Biplot: sobreposicion de individuos y variables")
    ax.set_xlabel(f"PC1 ({explained[0]:.2f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1]:.2f}% var)")
    ax.legend(title="Clase", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

