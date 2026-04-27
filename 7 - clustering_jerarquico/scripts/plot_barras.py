"""Graficos de barras: tamanos de cluster y perfiles medios por variable."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_cluster_sizes(labels: np.ndarray, *, ax: plt.Axes | None = None, title: str = "") -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    s = pd.Series(labels, name="cluster").value_counts().sort_index()
    colors = sns.color_palette("viridis", n_colors=len(s))
    ax.bar(s.index.astype(str), s.values, color=colors, edgecolor="0.2")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Cantidad de observaciones")
    ax.set_title(title or "Tamanos de cluster")
    for i, v in enumerate(s.values):
        ax.text(i, v + 0.05, str(int(v)), ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return ax


def plot_mean_profile_bars(
    df: pd.DataFrame,
    feature_cols: list[str],
    labels: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    title: str = "",
) -> plt.Axes:
    """Barras agrupadas: promedio por materia y cluster (escala original de calificaciones)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    tmp = df[feature_cols].copy()
    tmp["cluster"] = labels
    long = tmp.groupby("cluster", observed=True)[feature_cols].mean().reset_index().melt(
        id_vars="cluster", var_name="variable", value_name="promedio"
    )
    sns.barplot(data=long, x="variable", y="promedio", hue="cluster", ax=ax, palette="Set2")
    ax.set_xlabel("Asignatura")
    ax.set_ylabel("Promedio en la escala de datos")
    ax.set_title(title or "Perfil medio por cluster (barras)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    return ax
