"""Mapa de calor del perfil medio por cluster (variables en filas, clusters en columnas)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_cluster_profile_heatmap(
    df: pd.DataFrame,
    feature_cols: list[str],
    labels: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    title: str = "",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    tmp = df[feature_cols].copy()
    tmp["cluster"] = labels
    prof = tmp.groupby("cluster", observed=True)[feature_cols].mean().T
    sns.heatmap(prof, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax, linewidths=0.5)
    ax.set_title(title or "Promedios por cluster (mapa de calor)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Asignatura")
    plt.tight_layout()
    return ax
