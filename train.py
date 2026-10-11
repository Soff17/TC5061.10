"""
TC5061 – Experimentación con MLflow
Script de entrenamiento parametrizado para el dataset California Housing.

Migra el flujo del notebook baseline (carga -> limpieza IQR -> split ->
entrenamiento -> evaluación) a funciones con una sola responsabilidad y
registra cada corrida en MLflow Tracking.

Ejemplo:
    python train.py                                   # valores por defecto
    python train.py --n-estimators 300 --max-depth 20 --seed 42
    python train.py --model linear --run-name baseline_lineal
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin interfaz gráfica: el script corre en terminal
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

TARGET = "MedHouseVal"
OUTLIER_COLS = ["AveRooms", "AveBedrms", "Population", "AveOccup"]
DEFAULT_DATA = Path(__file__).parent / "data" / "california_housing.csv"


# --------------------------------------------------------------------------- #
# 1. Argumentos de línea de comandos
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un regresor sobre California Housing y registra la corrida en MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_model = parser.add_argument_group("Modelo")
    g_model.add_argument("--model", choices=["random_forest", "linear"], default="random_forest",
                         help="Tipo de modelo. 'linear' reproduce el baseline del notebook.")
    g_model.add_argument("--n-estimators", type=int, default=100,
                         help="[random_forest] Número de árboles del bosque.")
    g_model.add_argument("--max-depth", type=int, default=None,
                         help="[random_forest] Profundidad máxima de cada árbol (None = sin límite).")
    g_model.add_argument("--min-samples-leaf", type=int, default=1,
                         help="[random_forest] Mínimo de muestras por hoja; valores altos regularizan.")
    g_model.add_argument("--max-features", type=str, default="1.0",
                         help="[random_forest] Fracción (p. ej. 0.5) o 'sqrt'/'log2' de features por split.")

    g_data = parser.add_argument_group("Datos y preprocesamiento")
    g_data.add_argument("--data-path", type=Path, default=DEFAULT_DATA, help="Ruta al CSV del dataset.")
    g_data.add_argument("--test-size", type=float, default=0.2, help="Proporción del conjunto de prueba.")
    g_data.add_argument("--iqr-factor", type=float, default=3.0,
                        help="Multiplicador del IQR para eliminar outliers (<=0 desactiva la limpieza).")

    g_run = parser.add_argument_group("Reproducibilidad y MLflow")
    g_run.add_argument("--seed", type=int, default=42,
                       help="Semilla para split, modelo, numpy y random.")
    g_run.add_argument("--experiment-name", type=str, default="TC5061_CaliforniaHousing_RandomForest_Tuning",
                       help="Experimento de MLflow donde se agrupan las corridas.")
    g_run.add_argument("--run-name", type=str, default=None, help="Nombre legible de la corrida.")
    g_run.add_argument("--tracking-uri", type=str,
                       default=os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
                       help="Backend de MLflow Tracking.")

    args = parser.parse_args(argv)
    args.max_features = _parse_max_features(args.max_features)
    return args


def _parse_max_features(value: str):
    """Acepta 'sqrt'/'log2' o un número (fracción si es float, conteo si es entero)."""
    if value in {"sqrt", "log2"}:
        return value
    num = float(value)
    return int(num) if num.is_integer() and num > 1 else num


# --------------------------------------------------------------------------- #
# 2. Reproducibilidad
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Fija todas las fuentes de aleatoriedad usadas en el script."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def file_md5(path: Path) -> str:
    """Huella del archivo de datos, para saber con qué versión exacta se entrenó."""
    return hashlib.md5(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# 3. Datos
# --------------------------------------------------------------------------- #
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en {path}")
    return pd.read_csv(path)


def remove_outliers(df: pd.DataFrame, cols: list[str], factor: float) -> pd.DataFrame:
    """Elimina filas por encima de Q3 + factor*IQR en cada columna (mismo criterio del notebook)."""
    if factor <= 0:
        return df.copy()
    clean = df.copy()
    for col in cols:
        q1, q3 = clean[col].quantile([0.25, 0.75])
        clean = clean[clean[col] <= q3 + factor * (q3 - q1)]
    return clean


def split_data(df: pd.DataFrame, test_size: float, seed: int):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed)


# --------------------------------------------------------------------------- #
# 4. Modelo
# --------------------------------------------------------------------------- #
def build_model(args: argparse.Namespace):
    if args.model == "linear":
        return LinearRegression()
    return RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=args.seed,
        n_jobs=-1,
    )


def model_params(args: argparse.Namespace) -> dict:
    """Hiperparámetros que aplican al modelo elegido (para no registrar valores sin efecto)."""
    params = {"model": args.model}
    if args.model == "random_forest":
        params.update(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth is not None else "None",
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
        )
    return params


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


# --------------------------------------------------------------------------- #
# 5. Artefactos
# --------------------------------------------------------------------------- #
def plot_real_vs_pred(y_true, y_pred, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.3, s=8, color="steelblue")
    lims = [y_true.min(), y_true.max()]
    ax.plot(lims, lims, "r--", lw=1.5, label="Predicción perfecta")
    ax.set_xlabel("Valor real (cientos de miles USD)")
    ax.set_ylabel("Valor predicho")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_feature_importance(model, feature_names, path: Path) -> None:
    if hasattr(model, "feature_importances_"):
        values, label = model.feature_importances_, "Importancia (reducción de impureza)"
    else:
        values, label = model.coef_, "Coeficiente"
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(np.array(feature_names)[order], values[order], color="steelblue")
    ax.set_xlabel(label)
    ax.set_title("Importancia de variables")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def write_metrics_json(metrics: dict, params: dict, path: Path) -> None:
    path.write_text(json.dumps({"params": params, "metrics": metrics}, indent=2, default=str))


# --------------------------------------------------------------------------- #
# 6. Orquestación
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)

    df_raw = load_data(args.data_path)
    df = remove_outliers(df_raw, OUTLIER_COLS, args.iqr_factor)
    X_train, X_test, y_train, y_test = split_data(df, args.test_size, args.seed)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name) as run:
        # --- Parámetros ---------------------------------------------------- #
        params = model_params(args)
        params.update(
            seed=args.seed,
            test_size=args.test_size,
            iqr_factor=args.iqr_factor,
            rows_raw=len(df_raw),
            rows_clean=len(df),
            data_md5=file_md5(args.data_path),
        )
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.set_tag("dataset", "California Housing")

        # --- Entrenamiento y métricas -------------------------------------- #
        model = build_model(args)
        model.fit(X_train, y_train)

        test_metrics = evaluate(model, X_test, y_test)
        train_metrics = evaluate(model, X_train, y_train)
        metrics = {f"test_{k}": v for k, v in test_metrics.items()}
        metrics.update({f"train_{k}": v for k, v in train_metrics.items()})
        metrics["overfit_gap_rmse"] = metrics["test_rmse"] - metrics["train_rmse"]
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # --- Artefactos ---------------------------------------------------- #
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plot_real_vs_pred(y_test, model.predict(X_test), tmp / "real_vs_predicho.png",
                              f"Real vs predicho – {args.model}")
            plot_feature_importance(model, list(X_train.columns), tmp / "feature_importance.png")
            write_metrics_json(metrics, params, tmp / "metrics.json")
            mlflow.log_artifact(str(tmp / "real_vs_predicho.png"), artifact_path="plots")
            mlflow.log_artifact(str(tmp / "feature_importance.png"), artifact_path="plots")
            mlflow.log_artifact(str(tmp / "metrics.json"))

        # --- Modelo -------------------------------------------------------- #
        signature = infer_signature(X_test, model.predict(X_test))
        # MLflow 3.x serializa con skops; los árboles de sklearn deben declararse
        # explícitamente como tipo confiable (el modelo lo generamos nosotros).
        trusted = ["sklearn.tree._tree.Tree"] if args.model == "random_forest" else None
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_test.head(3),
            skops_trusted_types=trusted,
        )

        print(f"Run ID: {run.info.run_id}")
        print(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2))

    return metrics


if __name__ == "__main__":
    main()
