# 4-ACM — Análisis de correspondencias múltiples (Telco Churn)

Notebook `acm_telco_churn.ipynb`: descarga el dataset **Telco Customer Churn** desde una URL pública, conserva solo variables **cualitativas**, aplica **MCA** (`prince`) y compara visualmente con **PCA** sobre la tabla de indicadores (one-hot estandarizada).

## Entorno

```bash
cd "4-ACM"
pip install -r requirements.txt
jupyter nbconvert --execute acm_telco_churn.ipynb --to notebook --output acm_telco_churn.ipynb
```
