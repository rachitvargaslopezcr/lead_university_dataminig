import numpy as np
import pandas as pd
import prince


def fit_prince_pca(X_scaled_df: pd.DataFrame, n_components: int = 4, random_state: int = 42) -> prince.PCA:
    """Ajusta modelo PCA de Prince sobre dataframe estandarizado."""
    pca = prince.PCA(
        n_components=min(n_components, X_scaled_df.shape[1]),
        n_iter=5,
        rescale_with_mean=False,
        rescale_with_std=False,
        random_state=random_state,
    )
    return pca.fit(X_scaled_df)


def extract_pca_results(pca: prince.PCA, X_scaled_df: pd.DataFrame) -> dict[str, pd.DataFrame | np.ndarray]:
    """Extrae tablas base para graficar y analizar."""
    row_coords = pca.row_coordinates(X_scaled_df)
    row_coords.columns = [f"PC{i+1}" for i in range(row_coords.shape[1])]

    # Compatibilidad entre versiones de prince:
    # - algunas exponen `column_correlations` como metodo
    # - otras como atributo (DataFrame)
    col_corr_obj = pca.column_correlations
    col_corr = col_corr_obj(X_scaled_df) if callable(col_corr_obj) else col_corr_obj
    col_corr.columns = [f"PC{i+1}" for i in range(col_corr.shape[1])]

    eigenvalues = np.array(pca.eigenvalues_)
    explained = np.array(pca.percentage_of_variance_)
    cumulative = np.array(pca.cumulative_percentage_of_variance_)

    variance_df = pd.DataFrame(
        {
            "Componente": [f"PC{i+1}" for i in range(len(eigenvalues))],
            "Eigenvalor": eigenvalues,
            "% Varianza": explained,
            "% Acumulado": cumulative,
        }
    )

    return {
        "row_coords": row_coords,
        "col_corr": col_corr,
        "eigenvalues": eigenvalues,
        "explained": explained,
        "cumulative": cumulative,
        "variance_df": variance_df,
    }

