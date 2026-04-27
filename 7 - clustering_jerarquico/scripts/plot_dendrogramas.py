"""Dendrogramas para interpretar fusiones del clustering jerarquico."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram


def plot_dendrogram(
    Z: np.ndarray,
    leaf_labels: Sequence[str],
    *,
    ax: plt.Axes | None = None,
    title: str = "",
    color_threshold: float | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    dendrogram(
        Z,
        labels=list(leaf_labels),
        ax=ax,
        leaf_rotation=90,
        color_threshold=color_threshold,
        above_threshold_color="0.35",
    )
    ax.set_title(title)
    ax.set_xlabel("Observacion")
    ax.set_ylabel("Distancia de fusion")
    ax.grid(axis="y", alpha=0.25)
    fig = ax.figure
    if not getattr(fig, "get_constrained_layout", lambda: False)():
        fig.tight_layout()
    return ax
