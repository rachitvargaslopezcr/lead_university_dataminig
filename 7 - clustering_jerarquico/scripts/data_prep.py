from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_students_excel(
    path: str | Path | None = None,
) -> tuple[pd.DataFrame, str, list[str]]:
    """Carga Estudiantes.xlsx y devuelve el DataFrame, columna de id y columnas numericas."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "data" / "Estudiantes.xlsx"
    path = Path(path)
    df = pd.read_excel(path)
    name_col = "Nombre"
    if name_col not in df.columns:
        raise ValueError(f"Se esperaba columna '{name_col}'. Columnas: {list(df.columns)}")
    feature_cols = [c for c in df.columns if c != name_col]
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"La columna '{c}' debe ser numerica.")
    return df, name_col, feature_cols


def scale_features(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """Estandariza variables (media 0, varianza 1). Devuelve matriz X, scaler y DataFrame escalado."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].to_numpy(dtype=float))
    scaled_df = pd.DataFrame(X, columns=feature_cols, index=df.index)
    return X, scaler, scaled_df
