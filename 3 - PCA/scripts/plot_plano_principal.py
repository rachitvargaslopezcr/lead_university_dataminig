import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_plano_principal(
    row_coords: pd.DataFrame, classes: pd.Series, explained: np.ndarray, title: str = "Plano principal de individuos"
) -> None:
    """Grafica PC1 vs PC2 para individuos."""
    plot_df = row_coords.copy()
    plot_df["class"] = classes.values

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="class", s=70, alpha=0.85)
    plt.axhline(0, color="gray", linewidth=1)
    plt.axvline(0, color="gray", linewidth=1)
    plt.title(title)
    plt.xlabel(f"PC1 ({explained[0]:.2f}% var)")
    plt.ylabel(f"PC2 ({explained[1]:.2f}% var)")
    plt.legend(title="Clase", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

