"""Mapas de calor: matriz de dissimilaridad y version VAT (orden seriado)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import pdist, squareform


def _pdist_metric(metric: str) -> str:
    return "cityblock" if metric == "manhattan" else metric


def plot_distance_heatmap(
    X: np.ndarray,
    names: list[str],
    *,
    metric: str = "euclidean",
    ax: plt.Axes | None = None,
    title: str = "",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    D = squareform(pdist(X, metric=_pdist_metric(metric)))
    sns.heatmap(D, xticklabels=names, yticklabels=names, cmap="mako", ax=ax, square=True)
    ax.set_title(title or f"Matriz de distancias ({metric})")
    plt.tight_layout()
    return ax


def plot_vat_heatmap(
    D_ordered: np.ndarray,
    ordered_names: list[str],
    *,
    ax: plt.Axes | None = None,
    title: str = "",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(D_ordered, xticklabels=ordered_names, yticklabels=ordered_names, cmap="rocket", ax=ax, square=True)
    ax.set_title(title or "Matriz de distancias ordenada (VAT)")
    plt.tight_layout()
    return ax
