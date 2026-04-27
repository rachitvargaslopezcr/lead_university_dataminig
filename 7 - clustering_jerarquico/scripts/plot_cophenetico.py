"""Correlacion cophenetica: que tan bien el arbol jerarquico preserva las distancias originales."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

from hierarchical_clustering import ClusteringSpec, linkage_matrix, scipy_pdist_metric


def cophenetic_correlation(X: np.ndarray, spec: ClusteringSpec) -> float:
    Z = linkage_matrix(X, spec)
    if spec.linkage == "ward":
        condensed = pdist(X, metric="euclidean")
    else:
        condensed = pdist(X, metric=scipy_pdist_metric(spec.metric))
    corr, _ = cophenet(Z, condensed)
    return float(corr)


def plot_cophenetic_bars(
    X: np.ndarray,
    specs: list[ClusteringSpec],
    *,
    ax: plt.Axes | None = None,
    title: str = "",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    rows = [{"spec": s.label, "cophenetic": cophenetic_correlation(X, s)} for s in specs]
    df = pd.DataFrame(rows).sort_values("cophenetic", ascending=False)
    sns.barplot(data=df, y="spec", x="cophenetic", ax=ax, palette="crest")
    ax.set_xlabel("Correlacion cophenetica")
    ax.set_ylabel("")
    ax.set_title(title or "Calidad del arbol (cophenetica) por configuracion")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    return ax
