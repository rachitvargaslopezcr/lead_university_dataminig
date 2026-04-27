"""Proyeccion PCA 2D para visualizar grupos en espacio reducido."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def plot_pca_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    names: pd.Series | list[str],
    feature_cols: list[str],
    *,
    ax: plt.Axes | None = None,
    title: str = "",
    random_state: int = 42,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X)
    names_list = list(names)
    clusters = sorted(int(x) for x in np.unique(labels))
    cmap = plt.get_cmap("tab10")
    for j, c in enumerate(clusters):
        m = labels == c
        ax.scatter(
            coords[m, 0],
            coords[m, 1],
            c=[cmap(j % 10)],
            s=120,
            edgecolor="k",
            label=f"Cluster {c}",
        )
    for i, txt in enumerate(names_list):
        ax.annotate(txt, (coords[i, 0], coords[i, 1]), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
    ax.set_title(title or "PCA (2D) coloreada por cluster")
    ax.grid(alpha=0.25)
    ax.legend(title="Cluster", loc="best")
    plt.tight_layout()
    return ax
