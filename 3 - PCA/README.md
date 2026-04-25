# Proyecto PCA con Prince

Este proyecto implementa Analisis de Componentes Principales (PCA) usando la libreria `prince`.

## Estructura

- `main.ipynb`: notebook principal con todo el flujo de analisis.
- `scripts/`: modulos reutilizables por bloque de analisis y graficacion.
- `requirements.txt`: dependencias del proyecto.
- `data/`: carpeta para datasets propios (opcional).
- `outputs/`: carpeta para guardar graficos exportados (opcional).

## Incluye

- Preparacion y estandarizacion de datos.
- Ajuste del modelo PCA con `prince.PCA`.
- Plano principal (individuos).
- Grafico de correlacion (circulo de correlaciones).
- Varianza explicada (barras y acumulada).
- Grafico de sobreposicion (biplot).
- Contribucion de variables por componente.
- Cosenos al cuadrado (`cos2`) de variables.

## Ejecucion

1. Crear entorno virtual (opcional):
   - `python -m venv .venv`
   - `source .venv/bin/activate`
2. Instalar dependencias:
   - `pip install -r requirements.txt`
3. Abrir y ejecutar:
   - `jupyter notebook main.ipynb`

## Ejecucion por scripts

Si prefieres correr el flujo completo sin notebook:

- `python -m scripts.run_all`