"""Curva de silueta promedio vs numero de clusters k (para una o varias configuraciones)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from hierarchical_clustering import ClusteringSpec, silhouette_for_spec


def plot_silhouette_vs_k(
    X: np.ndarray,
    specs: list[ClusteringSpec],
    *,
    k_min: int = 2,
    k_max: int | None = None,
    ax: plt.Axes | None = None,
    title: str = "",
) -> plt.Axes:
    n = X.shape[0]
    if k_max is None:
        k_max = n - 1
    k_max = min(k_max, n - 1)
    rows: list[dict] = []
    for spec in specs:
        for k in range(k_min, k_max + 1):
            if k >= n:
                continue
            s = silhouette_for_spec(X, spec, k)
            rows.append({"k": k, "silhouette": s, "spec": spec.label})
    df = pd.DataFrame(rows)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=df, x="k", y="silhouette", hue="spec", marker="o", ax=ax)
    ax.set_xlabel("Numero de clusters (k)")
    ax.set_ylabel("Silueta promedio")
    ax.set_title(title or "Silueta vs k por configuracion")
    ax.grid(alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    return ax
