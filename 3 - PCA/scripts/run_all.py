"""
Script de ejemplo para ejecutar todo el flujo modular de PCA.
"""

from sklearn.datasets import load_iris

from scripts.data_prep import standardize_features
from scripts.pca_model import fit_prince_pca, extract_pca_results
from scripts.plot_plano_principal import plot_plano_principal
from scripts.plot_correlacion import plot_circulo_correlaciones
from scripts.plot_varianza import plot_varianza_explicada
from scripts.plot_sobreposicion import plot_biplot_sobreposicion
from scripts.metrics_contrib_cos2 import (
    compute_contribuciones,
    plot_contribuciones_pc1_pc2,
    compute_cos2_variables,
    plot_heatmap_cos2,
)


def main() -> None:
    data = load_iris(as_frame=True)
    df = data.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(data.target_names)))

    feature_cols = data.feature_names
    X_scaled_df, _ = standardize_features(df, feature_cols)
    y = df["species"]

    pca = fit_prince_pca(X_scaled_df, n_components=4, random_state=42)
    results = extract_pca_results(pca, X_scaled_df)

    row_coords = results["row_coords"]
    col_corr = results["col_corr"]
    explained = results["explained"]
    cumulative = results["cumulative"]
    eigenvalues = results["eigenvalues"]

    plot_plano_principal(row_coords, y, explained)
    plot_circulo_correlaciones(col_corr, explained)
    plot_varianza_explicada(explained, cumulative)
    plot_biplot_sobreposicion(row_coords, col_corr, y, explained)

    contrib_df = compute_contribuciones(col_corr, eigenvalues)
    print("\nContribuciones:\n", contrib_df)
    plot_contribuciones_pc1_pc2(contrib_df)

    cos2_df = compute_cos2_variables(col_corr)
    print("\nCos2:\n", cos2_df)
    plot_heatmap_cos2(cos2_df)


if __name__ == "__main__":
    main()

