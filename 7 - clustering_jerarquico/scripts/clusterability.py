"""Indicadores de tendencia a agrupar (clusterability) antes del clustering jerarquico."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist, squareform


def _pdist_metric(metric: str) -> str:
    if metric == "manhattan":
        return "cityblock"
    return metric


def hopkins_statistic(
    X: np.ndarray,
    m: int | None = None,
    *,
    seed: int = 42,
) -> float:
    """
    Estadistico de Hopkins (aprox. en [0, 1]).

    Valores cercanos a 0.5 sugieren datos similares a uniforme aleatorio (poca estructura).
    Valores mayores (p. ej. > 0.7) sugieren mayor tendencia a formar grupos compactos.
    Con pocas observaciones el valor es muy variable; usarlo como apoyo, no como veredicto unico.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    if n < 3:
        raise ValueError("Se necesitan al menos 3 observaciones para Hopkins.")
    if m is None:
        m = max(2, min(n - 1, max(2, n // 2)))
    m = int(m)
    m = max(2, min(m, n - 1))

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    u = np.empty(m, dtype=float)
    for i in range(m):
        j = int(rng.integers(0, n))
        row = X[j]
        dists = np.linalg.norm(X - row, axis=1)
        dists[j] = np.inf
        u[i] = float(dists.min())

    artificial = rng.uniform(low=X_min, high=X_max, size=(m, d))
    w = np.empty(m, dtype=float)
    for i in range(m):
        dists = np.linalg.norm(X - artificial[i], axis=1)
        w[i] = float(dists.min())

    denom = u.sum() + w.sum() + 1e-12
    return float(w.sum() / denom)


def vat_order(
    X: np.ndarray,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Orden VAT (Visual Assessment of Tendency): orden de filas/columnas por enlace simple
    sobre la matriz de distancias. Devuelve indices de orden y matriz D reordenada.
    """
    condensed = pdist(X, metric=_pdist_metric(metric))
    D = squareform(condensed)
    Z = linkage(condensed, method="single")
    order = np.asarray(leaves_list(Z), dtype=int)
    D_ord = D[np.ix_(order, order)]
    return order, D_ord
