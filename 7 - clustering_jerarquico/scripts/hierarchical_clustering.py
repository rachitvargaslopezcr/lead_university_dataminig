"""Clustering jerarquico: distancias, linkage y etiquetas con scipy + sklearn."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def scipy_pdist_metric(metric: str) -> str:
    """scipy.pdist usa 'cityblock' en lugar de 'manhattan'."""
    if metric == "manhattan":
        return "cityblock"
    return metric


@dataclass(frozen=True)
class ClusteringSpec:
    """Especificacion de una corrida: metrica de distancia y metodo de enlace."""

    metric: str
    linkage: str
    label: str

    def sklearn_metric(self) -> str:
        m = self.metric
        if m == "manhattan":
            return "manhattan"
        return m


def linkage_matrix(X: np.ndarray, spec: ClusteringSpec) -> np.ndarray:
    """Matriz Z de scipy (formato linkage) para dendrogramas y cophenetica."""
    if spec.linkage == "ward":
        if spec.metric != "euclidean":
            raise ValueError("Ward en scipy/sklearn usa distancia euclidea en el espacio de observaciones.")
        return linkage(X, method="ward", metric="euclidean")
    condensed = pdist(X, metric=scipy_pdist_metric(spec.metric))
    return linkage(condensed, method=spec.linkage)


def labels_sklearn(X: np.ndarray, spec: ClusteringSpec, n_clusters: int) -> np.ndarray:
    """Etiquetas de cluster usando AgglomerativeClustering (consistente con metricas sklearn)."""
    metric = spec.sklearn_metric()
    if spec.linkage == "ward":
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",
            metric="euclidean",
        )
    else:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=spec.linkage,
            metric=metric,
        )
    return model.fit_predict(X)


def labels_from_Z(Z: np.ndarray, n_clusters: int) -> np.ndarray:
    """Etiquetas a partir de Z usando criterio de numero maximo de grupos."""
    return fcluster(Z, t=n_clusters, criterion="maxclust")


def default_specs() -> list[ClusteringSpec]:
    """Combinaciones didacticas: varias metricas y single/complete/average/ward."""
    metrics = [
        ("euclidean", "euclidea"),
        ("manhattan", "Manhattan (L1)"),
        ("cosine", "coseno"),
    ]
    linkages = [
        ("single", "salto minimo (single)"),
        ("complete", "salto maximo (complete)"),
        ("average", "enlace promedio (average)"),
        ("ward", "Ward (min. varianza intra)"),
    ]
    specs: list[ClusteringSpec] = []
    for met, met_name in metrics:
        for lk, lk_name in linkages:
            if lk == "ward" and met != "euclidean":
                continue
            specs.append(
                ClusteringSpec(
                    metric=met,
                    linkage=lk,
                    label=f"{met_name} | {lk_name}",
                )
            )
    return specs


def silhouette_for_spec(
    X: np.ndarray,
    spec: ClusteringSpec,
    n_clusters: int,
) -> float:
    """Silhouette en el mismo espacio/metrica que el modelo cuando es posible."""
    labels = labels_sklearn(X, spec, n_clusters=n_clusters)
    if len(np.unique(labels)) < 2:
        return float("nan")
    metric = spec.sklearn_metric()
    if spec.linkage == "ward":
        return float(silhouette_score(X, labels, metric="euclidean"))
    return float(silhouette_score(X, labels, metric=metric))


def grid_silhouette(
    X: np.ndarray,
    specs: list[ClusteringSpec],
    k_min: int = 2,
    k_max: int | None = None,
) -> pd.DataFrame:
    """Tabla larga: spec, k, silhouette."""
    n = X.shape[0]
    if k_max is None:
        k_max = max(2, n - 1)
    k_max = min(k_max, n - 1)
    rows: list[dict] = []
    for spec in specs:
        for k in range(k_min, k_max + 1):
            if k >= n:
                continue
            s = silhouette_for_spec(X, spec, k)
            rows.append({"spec": spec.label, "metric": spec.metric, "linkage": spec.linkage, "k": k, "silhouette": s})
    return pd.DataFrame(rows)
