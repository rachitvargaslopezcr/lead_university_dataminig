import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def compute_contribuciones(col_corr: pd.DataFrame, eigenvalues: np.ndarray) -> pd.DataFrame:
    """Calcula contribuciones de variables por componente."""
    contrib = (col_corr**2).copy()
    for i, pc in enumerate(contrib.columns):
        contrib[pc] = (contrib[pc] / eigenvalues[i]) * 100
    return contrib.round(3)


def plot_contribuciones_pc1_pc2(contrib_df: pd.DataFrame) -> None:
    """Grafica contribuciones de variables para PC1 y PC2."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for idx, pc in enumerate(["PC1", "PC2"]):
        ordered = contrib_df[pc].sort_values(ascending=False)
        sns.barplot(x=ordered.values, y=ordered.index, ax=axes[idx], palette="viridis")
        axes[idx].set_title(f"Contribucion de variables en {pc}")
        axes[idx].set_xlabel("Contribucion (%)")
        axes[idx].set_ylabel("" if idx == 1 else "Variable")

    plt.tight_layout()
    plt.show()


def compute_cos2_variables(col_corr: pd.DataFrame) -> pd.DataFrame:
    """Calcula cosenos al cuadrado (cos2) de variables."""
    return (col_corr**2).round(4)


def plot_heatmap_cos2(cos2_df: pd.DataFrame) -> None:
    """Grafica heatmap de cos2 por variable y componente."""
    plt.figure(figsize=(8, 4))
    sns.heatmap(cos2_df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Cos2 de variables por componente")
    plt.xlabel("Componentes")
    plt.ylabel("Variables")
    plt.tight_layout()
    plt.show()

