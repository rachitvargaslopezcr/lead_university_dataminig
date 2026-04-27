"""Grafico de arana (radar) comparando perfiles medios de cada cluster."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_radar_profiles(
    df: pd.DataFrame,
    feature_cols: list[str],
    labels: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    title: str = "",
) -> plt.Axes:
    """
    Normaliza cada variable al rango [0, 1] sobre todos los datos para comparar formas
    entre clusters en la misma escala angular.
    """
    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
    vals = df[feature_cols].to_numpy(dtype=float)
    vmin = vals.min(axis=0)
    vmax = vals.max(axis=0)
    span = np.where(vmax - vmin < 1e-9, 1.0, vmax - vmin)
    normed = (vals - vmin) / span

    categories = list(feature_cols)
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    clusters = sorted(int(c) for c in np.unique(labels))
    cmap = plt.get_cmap("tab10")
    for j, c in enumerate(clusters):
        mask = labels == c
        profile = normed[mask].mean(axis=0)
        profile = np.concatenate([profile, profile[:1]])
        color = cmap(j % 10)
        ax.plot(angles, profile, "o-", linewidth=2, label=f"Cluster {c}", color=color)
        ax.fill(angles, profile, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title(title or "Perfil tipo arana (variables normalizadas 0-1)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))
    plt.tight_layout()
    return ax
