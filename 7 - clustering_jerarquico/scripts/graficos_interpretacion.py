"""Gráficos de interpretación y exploración para clustering jerárquico (importar desde el notebook)."""

from __future__ import annotations

from math import pi
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score


def plot_matriz_distancias(X: np.ndarray, names: list[str], ax, title: str) -> None:
    D = squareform(pdist(X, metric="euclidean"))
    sns.heatmap(D, xticklabels=names, yticklabels=names, cmap="mako", ax=ax, square=True)
    ax.set_title(title)


def plot_matriz_vat(D_ord: np.ndarray, names_ord: list[str], ax, title: str) -> None:
    sns.heatmap(D_ord, xticklabels=names_ord, yticklabels=names_ord, cmap="rocket", ax=ax, square=True)
    ax.set_title(title)


def plot_dendrogramas_cuadricula(
    X: np.ndarray,
    names: list[str],
    methods: list[tuple[str, str]],
    linkage_fn: Callable[[np.ndarray, str], np.ndarray],
    *,
    figsize: tuple[float, float] = (14, 9),
    suptitle: str | None = None,
    xlabel: str = "Observación",
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes = axes.ravel()
    for ax, (met, title) in zip(axes, methods):
        Z = linkage_fn(X, met)
        dendrogram(Z, labels=names, ax=ax, leaf_rotation=90, color_threshold=0, above_threshold_color="0.35")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Altura de fusión")
    if suptitle:
        fig.suptitle(suptitle, y=1.02, fontsize=13)
    return fig


def plot_silueta_vs_k(sil_df: pd.DataFrame, ax, K: int, title: str = "Silueta media vs k (Ward + euclídea)") -> None:
    sns.lineplot(data=sil_df, x="k", y="silhouette", marker="o", ax=ax)
    ax.set_title(title)
    ax.axvline(K, color="crimson", ls="--", label=f"k = {K}")
    ax.legend()


def plot_barras_cophenetica(coph_vals: list[float], labels_c: list[str], ax) -> None:
    sns.barplot(x=coph_vals, y=labels_c, ax=ax, orient="h", hue=labels_c, legend=False, palette="crest")
    ax.set_xlabel("Correlación cophenética")
    ax.set_title("Calidad del árbol (cophenética)")
    ax.set_xlim(0, 1)


def plot_tamanos_cluster(sizes: pd.Series, ax, K: int, ylabel: str = "Número de observaciones") -> None:
    sns.barplot(
        x=sizes.index.astype(str),
        y=sizes.values,
        ax=ax,
        hue=sizes.index.astype(str),
        legend=False,
        palette="viridis",
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Tamaños de cluster (k = {K})")
    for i, v in enumerate(sizes.values):
        ax.text(i, v + 0.05 * max(sizes.values.max(), 1), str(int(v)), ha="center", fontsize=10)


def plot_perfil_barras_z(melted: pd.DataFrame, ax, title: str = "Perfil medio por cluster (datos estandarizados)") -> None:
    sns.barplot(data=melted, x="Variable", y="Media (z)", hue="cluster", ax=ax, palette="Set2")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_title(title)


def plot_radar_centros(centers: np.ndarray, feature_names: list[str], *, figsize: tuple[float, float] = (7, 7)) -> plt.Figure:
    n_clusters, n_feat = centers.shape
    normed = []
    for j in range(n_feat):
        col = centers[:, j]
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-9:
            normed.append(np.full(n_clusters, 50.0))
        else:
            normed.append((col - lo) / (hi - lo) * 100.0)
    M = np.array(normed).T
    angles = [i / float(n_feat) * 2 * pi for i in range(n_feat)]
    angles += angles[:1]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(list(feature_names), fontsize=9)
    cmap = plt.get_cmap("tab10")
    for c in range(n_clusters):
        vals = M[c].tolist() + M[c].tolist()[:1]
        ax.plot(angles, vals, "o-", label=f"Cluster {c}", color=cmap(c % 10))
        ax.fill(angles, vals, alpha=0.12, color=cmap(c % 10))
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))
    ax.set_title("Perfil medio por cluster (escala 0–100 por variable)")
    fig.tight_layout()
    return fig


def plot_mapa_calor_medias(pivot: pd.DataFrame, ax, title: str = "Medias por cluster (z-score)") -> None:
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title(title)


def plot_pca_2d(
    X: np.ndarray,
    names: list[str],
    labels: np.ndarray,
    *,
    random_state: int = 42,
    figsize: tuple[float, float] = (8, 6),
) -> tuple[plt.Figure, PCA]:
    pca = PCA(n_components=2, random_state=random_state)
    xy = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for c in sorted(np.unique(labels)):
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=85, label=f"Cluster {c}", edgecolor="k", alpha=0.9)
    for i, txt in enumerate(names):
        ax.annotate(txt, (xy[i, 0], xy[i, 1]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f} % var.)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f} % var.)")
    ax.set_title("PCA (2D) coloreada por cluster")
    ax.legend(title="Cluster")
    ax.grid(alpha=0.3)
    return fig, pca


def plot_silueta_por_observacion(
    sil_vals: np.ndarray,
    names: list[str],
    X: np.ndarray,
    labels: np.ndarray,
    *,
    K: int,
    figsize: tuple[float, float] = (9, 5),
    titulo_obs: str = "estudiante",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    y_pos = np.arange(len(names))
    order = np.argsort(sil_vals)
    vmin, vmax = float(sil_vals.min()), float(sil_vals.max())
    if vmax - vmin < 1e-12:
        norm_fn = lambda v: 0.5
    elif vmin < 0 < vmax:
        norm_fn = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm_fn = Normalize(vmin=vmin, vmax=vmax)
    colors = [plt.cm.RdYlGn(norm_fn(v)) for v in sil_vals[order]]
    ax.barh(y_pos, sil_vals[order], color=colors, edgecolor="0.35", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(np.array(names)[order], fontsize=8)
    ax.axvline(silhouette_score(X, labels, metric="euclidean"), color="navy", ls="--", label="Media")
    ax.set_xlabel("Coeficiente de silueta")
    ax.set_title(f"Silueta por {titulo_obs} (Ward, k = {K})")
    ax.legend()
    return fig


def plot_dendrograma_ward_corte(
    Z: np.ndarray,
    names: list[str],
    k: int,
    *,
    figsize: tuple[float, float] = (12, 4.5),
    title: str | None = None,
) -> plt.Figure:
    n = len(names)
    heights = np.sort(Z[:, 2])
    idx = max(0, min(n - k - 1, len(heights) - 1))
    color_threshold = float(heights[idx]) if len(heights) else 0.0
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    dendrogram(Z, labels=names, ax=ax, leaf_rotation=90, color_threshold=color_threshold, above_threshold_color="0.45")
    ax.axhline(color_threshold, color="crimson", ls="--", lw=1.2, label=f"Corte aprox. (k ≈ {k})")
    ax.set_ylabel("Altura de fusión")
    ax.set_xlabel("Observación")
    ax.set_title(title or f"Dendrograma Ward con colores por corte (k ≈ {k})")
    ax.legend(loc="upper right")
    return fig


def plot_boxplots_por_cluster(
    df_num: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
    *,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    plot_df = df_num[feature_cols].copy()
    plot_df["cluster"] = labels.astype(int)
    melted = plot_df.melt(id_vars="cluster", var_name="variable", value_name="valor")
    n_vars = len(feature_cols)
    ncols = min(3, n_vars)
    nrows = int(np.ceil(n_vars / ncols))
    if figsize is None:
        figsize = (4 * ncols, 2.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True, squeeze=False)
    axes = np.atleast_2d(axes)
    for i, var in enumerate(feature_cols):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        sns.boxplot(data=melted[melted["variable"] == var], x="cluster", y="valor", ax=ax, palette="pastel")
        ax.set_title(var)
        ax.set_xlabel("Cluster")
    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].set_visible(False)
    fig.suptitle("Distribución por variable y cluster (escala original)", fontsize=12)
    return fig


def plot_violin_por_variable(
    df_num: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
    *,
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    plot_df = df_num[feature_cols].copy()
    plot_df["cluster"] = labels.astype(int)
    melted = plot_df.melt(id_vars="cluster", var_name="variable", value_name="valor")
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    sns.violinplot(data=melted, x="variable", y="valor", hue="cluster", ax=ax, palette="Set2", cut=0, inner="quart")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title("Violín: forma de la distribución por variable y cluster")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    return fig


def plot_perfil_lineas(pivot: pd.DataFrame, *, figsize: tuple[float, float] = (8, 4.5)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    x = np.arange(len(pivot.columns))
    for idx, row in pivot.iterrows():
        ax.plot(x, row.values, "o-", label=f"Cluster {idx}", linewidth=2, markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_ylabel("Media (z-score)")
    ax.set_title("Perfil medio: líneas por cluster")
    ax.axhline(0, color="0.5", ls=":", lw=1)
    ax.legend(title="Cluster")
    ax.grid(alpha=0.3)
    return fig


def plot_hist_silueta(sil_vals: np.ndarray, *, figsize: tuple[float, float] = (6, 3.5)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.hist(sil_vals, bins=min(12, max(4, len(sil_vals))), color="steelblue", edgecolor="0.2", alpha=0.85)
    ax.axvline(float(np.mean(sil_vals)), color="crimson", ls="--", label=f"Media = {np.mean(sil_vals):.3f}")
    ax.set_xlabel("Coeficiente de silueta")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de la silueta por observación")
    ax.legend()
    return fig


def plot_correlacion_variables(df_num: pd.DataFrame, feature_cols: list[str], *, figsize: tuple[float, float] = (6, 5)) -> plt.Figure:
    C = df_num[feature_cols].corr()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    sns.heatmap(C, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax, square=True)
    ax.set_title("Correlación entre variables (datos en escala del cuaderno)")
    return fig


def plot_pair_pca_extra(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    random_state: int = 42,
    figsize: tuple[float, float] = (9, 4),
) -> plt.Figure:
    pca3 = PCA(n_components=3, random_state=random_state)
    t = pca3.fit_transform(X)
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    for c in sorted(np.unique(labels)):
        m = labels == c
        axes[0].scatter(t[m, 0], t[m, 2], label=f"C{c}", s=70, edgecolor="k", alpha=0.9)
    axes[0].set_xlabel(f"PC1 ({pca3.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC3 ({pca3.explained_variance_ratio_[2]*100:.1f}%)")
    axes[0].set_title("PC1 vs PC3")
    axes[0].legend(title="Cluster", fontsize=8)
    axes[0].grid(alpha=0.3)
    for c in sorted(np.unique(labels)):
        m = labels == c
        axes[1].scatter(t[m, 1], t[m, 2], label=f"C{c}", s=70, edgecolor="k", alpha=0.9)
    axes[1].set_xlabel(f"PC2 ({pca3.explained_variance_ratio_[1]*100:.1f}%)")
    axes[1].set_ylabel(f"PC3 ({pca3.explained_variance_ratio_[2]*100:.1f}%)")
    axes[1].set_title("PC2 vs PC3")
    axes[1].legend(title="Cluster", fontsize=8)
    axes[1].grid(alpha=0.3)
    return fig


def plot_barras_apiladas_medias(pivot: pd.DataFrame, *, figsize: tuple[float, float] = (8, 4)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    x = np.arange(len(pivot.columns))
    cmap = plt.get_cmap("Set2")
    bottom = np.zeros(len(pivot.columns))
    for i, (cl, row) in enumerate(pivot.iterrows()):
        vals = row.values.astype(float)
        ax.bar(x, vals, bottom=bottom, label=f"Cluster {cl}", color=cmap(i % 8), edgecolor="0.25", width=0.72)
        bottom = bottom + vals
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Media (z) apilada por cluster")
    ax.set_title("Medias por cluster (barras apiladas)")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    return fig
