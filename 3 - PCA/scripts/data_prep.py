import pandas as pd
from sklearn.preprocessing import StandardScaler


def standardize_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    """Estandariza columnas numericas para PCA."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    scaled_df = pd.DataFrame(scaled, columns=feature_cols, index=df.index)
    return scaled_df, scaler

