"""Diagrama de coeficientes de silueta por observacion."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score


def plot_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str = "euclidean",
    ax: plt.Axes | None = None,
    title: str = "",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    if len(np.unique(labels)) < 2:
        ax.text(0.5, 0.5, "Se necesitan al menos 2 clusters", ha="center", va="center")
        return ax
    score = silhouette_score(X, labels, metric=metric)
    sample_sil = silhouette_samples(X, labels, metric=metric)
    y_lower = 10
    clusters = np.unique(labels)
    cmap = plt.get_cmap("tab10")
    for idx, c in enumerate(sorted(clusters.astype(int))):
        c_sil = sample_sil[labels == c]
        c_sil.sort()
        size = c_sil.shape[0]
        y_upper = y_lower + size
        color = cmap(idx % 10)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, c_sil, facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, f"c={c}")
        y_lower = y_upper + 10
    ax.axvline(score, color="red", linestyle="--", label=f"Promedio = {score:.3f}")
    ax.set_title(title or f"Silhouette (metrica={metric})")
    ax.set_xlabel("Coeficiente de silueta")
    ax.set_ylabel("Observaciones ordenadas por cluster")
    ax.legend(loc="best")
    ax.set_xlim(-0.2, 1.0)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    return ax
