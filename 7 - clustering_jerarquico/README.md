# 7 — Clustering jerarquico (didactico)

Flujo con datos `data/Estudiantes.xlsx`: validacion de clusterabilidad, comparacion de metricas de distancia y metodos de enlace, y graficos de interpretacion en `scripts/`.

## Entorno

```bash
cd "7 - clustering_jerarquico"
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Para abrir el notebook puedes usar Jupyter en tu entorno habitual (`pip install jupyter` o la extension de VS Code / Cursor).

Abre `clustering_jerarquico.ipynb` y ejecuta todas las celdas.

Tambien se incluye `clustering_jerarquico.executed.ipynb` (misma logica, ya corrida con `nbconvert`) para revisar salidas sin ejecutar.

## Contenido

- **clusterability**: estadistico de Hopkins y mapa tipo VAT (orden seriado de la matriz de distancias).
- **Modelo**: euclidea, Manhattan y coseno; enlace single (minimo), complete (maximo), average y Ward (solo con euclidea).
- **Graficos** (modulos en `scripts/`): dendrogramas, barras de tamanos y perfiles, PCA 2D, arana/radar, silueta, mapas de calor de distancias y perfiles, correlacion cophenetica, curva silueta vs k.

Con solo 10 estudiantes, los indicadores (sobre todo Hopkins y silueta) son inestables; el objetivo es didactico: interpretar formas del dendrograma y comparar configuraciones.
