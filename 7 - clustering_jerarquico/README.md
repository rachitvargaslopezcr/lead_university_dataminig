# 7 — Clustering jerárquico (cuaderno único)

Todo el flujo didáctico vive en **`clustering_jerarquico.ipynb`**: no hay módulos `.py` en esta carpeta. El ejemplo usa **`data/Estudiantes.xlsx`** (columna `Nombre` + calificaciones numéricas).

## Entorno

```bash
cd "7 - clustering_jerarquico"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Abre el notebook en Jupyter o en Cursor y ejecuta las celdas en orden (o *Run All*).

## Contenido del cuaderno

Hopkins y VAT, estandarización, dendrogramas (single, complete, average, Ward), silueta vs *k*, correlación cophenética, perfiles en barras y radar, mapa de calor de medias, PCA 2D, silueta por observación y tabla resumen. Puedes cambiar **`K`** en la sección de número de clusters.
