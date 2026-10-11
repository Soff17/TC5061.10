# TC5061 – Experimentación con MLflow (California Housing)

**Alumna:** Sofía Donlucas Bañuelos · A01655565

Migración del notebook baseline a un script de entrenamiento parametrizado (`train.py`) e instrumentado con **MLflow Tracking**. Cada ejecución registra sus parámetros, métricas, artefactos y el modelo entrenado, agrupados en el experimento `TC5061_CaliforniaHousing_RandomForest_Tuning`.

## Estructura

```
.
├── train.py                 # carga → limpieza IQR → split → entrenamiento → evaluación → registro MLflow
├── run_experiments.sh       # las 9 corridas reportadas
├── requirements.txt         # dependencias con versiones fijas
├── data/california_housing.csv
├── notebooks/01_exploracion_california_housing.ipynb   # solo EDA y decisiones de limpieza
└── docs/                    # capturas de la UI de MLflow
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Probado con Python 3.13. Con Python 3.11–3.12 también funciona si pip resuelve las mismas versiones.

## Ejecución

```bash
# Valores por defecto (Random Forest, 100 árboles, seed 42)
python train.py

# Cambiar hiperparámetros sin editar código
python train.py --n-estimators 300 --max-features 0.5 --min-samples-leaf 2 --seed 42 --run-name rf_combinada

# Reproducir el baseline lineal del notebook
python train.py --model linear --run-name baseline_lineal

# Todas las corridas del reporte
./run_experiments.sh

# Ver la ayuda completa
python train.py --help
```

### Argumentos

| Argumento | Default | Descripción |
|---|---|---|
| `--model` | `random_forest` | `random_forest` o `linear` (baseline del notebook) |
| `--n-estimators` | `100` | Número de árboles |
| `--max-depth` | `None` | Profundidad máxima de cada árbol |
| `--min-samples-leaf` | `1` | Mínimo de muestras por hoja (regularización) |
| `--max-features` | `1.0` | Fracción de variables por split, o `sqrt` / `log2` |
| `--seed` | `42` | Semilla para split, modelo, `numpy` y `random` |
| `--test-size` | `0.2` | Proporción del conjunto de prueba |
| `--iqr-factor` | `3.0` | Multiplicador IQR para eliminar outliers (`<=0` desactiva) |
| `--experiment-name` | `TC5061_CaliforniaHousing_RandomForest_Tuning` | Experimento de MLflow |
| `--run-name` | — | Nombre de la corrida |
| `--tracking-uri` | `sqlite:///mlflow.db` | Backend de tracking (o variable `MLFLOW_TRACKING_URI`) |

## Ver resultados en MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abre http://127.0.0.1:5000, cambia a la pestaña **Model training** y entra al experimento. Para comparar, selecciona las corridas y pulsa **Compare**.

## Qué se registra en cada corrida

- **Parámetros:** hiperparámetros del modelo, `seed`, `test_size`, `iqr_factor`, filas antes y después de la limpieza y el `data_md5` del CSV.
- **Métricas:** `test_rmse`, `test_mae`, `test_r2`, `train_rmse`, `train_mae`, `train_r2` y `overfit_gap_rmse` (test − train).
- **Artefactos:** `plots/real_vs_predicho.png`, `plots/feature_importance.png` y `metrics.json`.
- **Modelo:** `mlflow.sklearn.log_model` con firma e `input_example`.

## Reproducibilidad

- La semilla (`--seed`) controla el `train_test_split`, el `random_state` del Random Forest, `numpy.random` y `random`. Dos ejecuciones con los mismos argumentos producen métricas idénticas. Las corridas `02_rf_default` y `08_rf_default_repeticion` lo demuestran: ambas dan `test_rmse = 0.510874`.
- `requirements.txt` fija las versiones exactas usadas.
- El CSV va incluido en `data/` y su hash (`ca5b757f0729c68d74d05bf9d9ca8486`) queda registrado en cada corrida. Así se sabe con qué versión exacta de los datos se entrenó.

## Nota sobre el modelo

`LinearRegression`, el modelo del notebook, no tiene hiperparámetros que ajustar. Por eso el script lo conserva como baseline (`--model linear`) y usa un **Random Forest** para la experimentación. El Random Forest expone cuatro hiperparámetros relevantes.
