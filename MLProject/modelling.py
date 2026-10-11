import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


# Configuration 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

DEFAULT_DATA_FILE = os.path.join(
    "data", "processed", "harga_emas_dan_swiss_franc_preprocessing.csv"
)
DEFAULT_RAW_PRICE_FILE = os.path.join(
    "data", "raw", "XAU_USD Historical Data.csv"
)
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000/"

EXPERIMENT_NAME = "workflow-ci-random-forest-gold"

DATE_COLUMN = "Date"
TARGET_COLUMN = "return_gold"
FEATURE_COLUMNS = [
    "return_gold_lag_1",
    "return_gold_lag_2",
    "return_gold_lag_3",
    "return_gold_lag_4",
    "return_gold_lag_5",
    "return_gold_lag_6",
    "return_gold_lag_7",
    "return_gold_lag_8",
    "return_gold_lag_9",
    "return_gold_lag_10",
    "return_chf_lag_1",
    "return_chf_lag_2",
    "return_chf_lag_3",
    "return_chf_lag_4",
    "return_chf_lag_5",
    "return_chf_lag_6",
    "return_chf_lag_7",
    "return_chf_lag_8",
    "return_chf_lag_9",
    "return_chf_lag_10",
]

# Random Forest Hyperparameter
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = None

SPLIT_STRATEGY = "chronological"

MODEL_ARTIFACT_PATH = "model"
RUN_ID_FILENAME = "mlflow_run_id.txt"
METRICS_FILENAME = "metrics.json"
ARTIFACTS_SUBDIR = "artifacts"

PRICE_PLOT_FILENAME = "actual_vs_predicted_price.png"
PRICE_RESIDUAL_FILENAME = "residual_plot_price.png"
TEST_PREDICTIONS_FILENAME = "test_predictions.csv"

RAW_PRICE_DATE_COLUMN = "Date"
RAW_PRICE_VALUE_COLUMN = "Price"
RAW_PRICE_DATE_FORMAT = "%m/%d/%Y"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pelatihan ulang RandomForestRegressor untuk return_gold."
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_FILE,
        help="Path dataset hasil preprocessing (default: %(default)s).",
    )
    parser.add_argument(
        "--raw-price-path",
        default=DEFAULT_RAW_PRICE_FILE,
        help=(
            "Path dataset harga emas mentah untuk evaluasi tingkat harga "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Direktori penyimpanan run id, metrik, dan artefak (default: %(default)s).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=N_ESTIMATORS,
        help="Jumlah pohon RandomForest (default: %(default)s).",
    )
    return parser.parse_args(argv)


def _resolve_path(path: str, extra_dirs=()) -> str:
    """Cari file pada beberapa kandidat lokasi agar konsisten lintas CWD."""
    expanded = os.path.expanduser(path)
    candidates = [
        expanded,
        os.path.join(os.getcwd(), expanded),
        os.path.join(BASE_DIR, expanded),
        os.path.join(REPO_ROOT, expanded),
    ]
    for directory in extra_dirs:
        candidates.append(os.path.join(directory, os.path.basename(expanded)))
    candidates.append(os.path.join(REPO_ROOT, os.path.basename(expanded)))
    candidates.append(os.path.join(os.getcwd(), os.path.basename(expanded)))

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(
        "File tidak ditemukan. Lokasi yang dicoba: "
        + ", ".join(sorted({os.path.abspath(c) for c in candidates if c}))
    )


def resolve_data_path(path: str) -> str:
    return _resolve_path(path, extra_dirs=[os.path.join(REPO_ROOT, "data", "processed")])


def resolve_raw_price_path(path: str) -> str:
    return _resolve_path(path, extra_dirs=[os.path.join(REPO_ROOT, "data", "raw")])


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    df = pd.read_csv(path)

    required = [DATE_COLUMN, TARGET_COLUMN] + FEATURE_COLUMNS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan pada dataset: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Dataset mengandung nilai kosong pada kolom yang dibutuhkan.")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame):
    n_train = int(len(df) * (1.0 - TEST_SIZE))

    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    return train_df, test_df, x_train, y_train, x_test, y_test


# Evaluation
def evaluate(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mse ** 0.5),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def print_metrics(label: str, metrics: dict) -> None:
    print(f"\n===== Metrik {label} =====")
    for name in ["MAE", "RMSE", "MAPE", "R2"]:
        print(f"{name:>5}: {metrics[name]:.6f}")


# Load Raw Prices
def load_raw_prices(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File harga mentah tidak ditemukan: {path}")

    raw = pd.read_csv(path)
    required = [RAW_PRICE_DATE_COLUMN, RAW_PRICE_VALUE_COLUMN]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"Kolom harga mentah tidak ditemukan: {missing}")

    raw = raw[[RAW_PRICE_DATE_COLUMN, RAW_PRICE_VALUE_COLUMN]].copy()
    raw["Date"] = pd.to_datetime(
        raw[RAW_PRICE_DATE_COLUMN], format=RAW_PRICE_DATE_FORMAT
    )
    raw["Price"] = (
        raw[RAW_PRICE_VALUE_COLUMN]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    raw = (
        raw[["Date", "Price"]]
        .drop_duplicates("Date")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    return raw


def validate_returns(df: pd.DataFrame, raw: pd.DataFrame) -> float:
    """Memastikan return_gold dataset konsisten dengan pct_change harga mentah."""
    raw_ret = raw.assign(ret=raw["Price"].pct_change()).set_index("Date")["ret"]
    recomputed = df["Date"].map(raw_ret)
    max_diff = float((df[TARGET_COLUMN] - recomputed).abs().max())
    print(f"Validasi return vs harga mentah: max abs diff = {max_diff:.3e}")
    if max_diff > 1e-6:
        raise ValueError(
            "return_gold pada dataset tidak cocok dengan harga mentah. Periksa kolom "
            "tanggal/harga atau format tanggal pada data harga mentah."
        )
    return max_diff


def attach_prices(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["prev_price"] = raw["Price"].shift(1)
    price_map = raw.set_index("Date")

    df = df.copy()
    df["actual_price"] = df["Date"].map(price_map["Price"])
    df["prev_price"] = df["Date"].map(price_map["prev_price"])

    if df["actual_price"].isna().any() or df["prev_price"].isna().any():
        raise ValueError("Harga mentah tidak mencakup seluruh tanggal dataset.")
    return df


def reconstruct_predicted_prices(prev_price, y_pred_return):
    """Rekonstruksi return prediksi menjadi harga via P_t = P_{t-1} * (1 + r_t)."""
    return prev_price.to_numpy() * (1.0 + y_pred_return)


# Artifact
def save_actual_vs_predicted(dates, y_true, y_pred, path, title, ylabel):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label="Aktual", linewidth=1)
    plt.plot(dates, y_pred, label="Prediksi", linewidth=1)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def save_residual_plot(dates, y_true, y_pred, path, title, ylabel):
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 5))
    plt.scatter(dates, residuals, s=6, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def ensure_experiment(name: str) -> None:
    """Buat experiment bila belum ada dengan artifact root absolut."""
    existing = mlflow.get_experiment_by_name(name)
    if existing is not None:
        return
    artifact_root = Path(BASE_DIR, "mlruns").resolve().as_uri()
    mlflow.create_experiment(name, artifact_location=artifact_root)


def log_model(model) -> str:
    params = inspect.signature(mlflow.sklearn.log_model).parameters
    kwargs = {}
    if "skops_trusted_types" in params:

        kwargs["skops_trusted_types"] = ["sklearn.tree._tree.Tree"]
    if "name" in params:
        return mlflow.sklearn.log_model(model, name=MODEL_ARTIFACT_PATH, **kwargs)
    return mlflow.sklearn.log_model(model, artifact_path=MODEL_ARTIFACT_PATH, **kwargs)


def write_outputs(output_dir: str, run_id: str, summary: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)

    run_id_path = os.path.join(output_dir, RUN_ID_FILENAME)
    with open(run_id_path, "w", encoding="utf-8") as handle:
        handle.write(run_id + "\n")

    metrics_path = os.path.join(output_dir, METRICS_FILENAME)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Run ID disimpan di : {run_id_path}")
    print(f"Metrik disimpan di : {metrics_path}")


def download_model_artifact(run_id: str, output_dir: str) -> str:
    """Unduh artefak model dari run terbaru ke output_dir/model."""
    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}"
    local_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=output_dir)
    print(f"Model artefak ({model_uri}) tersalin ke: {local_path}")
    return local_path


# Main
def main(argv=None) -> None:
    args = parse_args(argv)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)

    data_path = resolve_data_path(args.data_path)
    output_dir = os.path.abspath(args.output_dir)
    artifacts_dir = os.path.join(output_dir, ARTIFACTS_SUBDIR)
    os.makedirs(artifacts_dir, exist_ok=True)

    running_via_project = bool(os.environ.get("MLFLOW_RUN_ID"))
    if not running_via_project:
        ensure_experiment(EXPERIMENT_NAME)
        mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_dataset(data_path)

    try:
        raw_price_path = resolve_raw_price_path(args.raw_price_path)
        raw = load_raw_prices(raw_price_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Data harga mentah (harga emas USD) wajib tersedia untuk evaluasi "
            "tingkat harga. "
            f"{exc}"
        ) from exc

    validate_returns(df, raw)
    df = attach_prices(df, raw)

    train_df, test_df, x_train, y_train, x_test, y_test = chronological_split(df)

    print(f"Tracking URI         : {tracking_uri}")
    print(f"Dataset              : {data_path}")
    print(f"Data harga mentah    : {raw_price_path}")
    print(f"Jumlah baris         : {len(df)}")
    print(
        f"Rentang tanggal      : {df[DATE_COLUMN].min().date()} "
        f"s.d. {df[DATE_COLUMN].max().date()}"
    )
    print(f"Fitur                : {FEATURE_COLUMNS}")
    print(f"Target               : {TARGET_COLUMN}")
    print(f"Data training/testing: {len(x_train)} / {len(x_test)}")
    print(f"Evaluasi harga USD   : aktif")

    params = {
        "model": "RandomForestRegressor",
        "n_estimators": int(args.n_estimators),
        "max_depth": str(MAX_DEPTH) if MAX_DEPTH is not None else "None",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "split_strategy": SPLIT_STRATEGY,
        "target": TARGET_COLUMN,
        "n_features": len(FEATURE_COLUMNS),
        "feature_names": ",".join(FEATURE_COLUMNS),
        "n_train": len(x_train),
        "n_test": len(x_test),
    }

    model = RandomForestRegressor(
        n_estimators=int(args.n_estimators),
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="rf_retrain") as run:
        run_id = run.info.run_id

        mlflow.log_params(params)

        model.fit(x_train, y_train)
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

        logged_metrics = {}

        summary = {
            "run_id": run_id,
            "experiment_id": run.info.experiment_id,
            "experiment_name": mlflow.get_experiment(run.info.experiment_id).name,
            "tracking_uri": tracking_uri,
            "data_path": data_path,
            "target": TARGET_COLUMN,
            "features": FEATURE_COLUMNS,
            "n_estimators": int(args.n_estimators),
            "max_depth": str(MAX_DEPTH) if MAX_DEPTH is not None else "None",
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "split_strategy": SPLIT_STRATEGY,
            "n_train": len(x_train),
            "n_test": len(x_test),
            "predictions": {},
        }

        artifact_paths = []

        test_predictions = pd.DataFrame(
            {
                "Date": test_df[DATE_COLUMN].to_numpy(),
                "actual_return": y_test.to_numpy(),
                "predicted_return": y_pred_test,
            }
        )

        train_price_pred = reconstruct_predicted_prices(
            train_df["prev_price"], y_pred_train
        )
        train_price_true = train_df["actual_price"].to_numpy()
        train_price_metrics = evaluate(train_price_true, train_price_pred)

        test_price_pred = reconstruct_predicted_prices(
            test_df["prev_price"], y_pred_test
        )
        test_price_true = test_df["actual_price"].to_numpy()
        test_price_metrics = evaluate(test_price_true, test_price_pred)

        print_metrics("Training (Harga USD)", train_price_metrics)
        print_metrics("Testing (Harga USD)", test_price_metrics)

        logged_metrics.update(
            {f"train_price_{k}": v for k, v in train_price_metrics.items()}
        )
        logged_metrics.update(
            {f"test_price_{k}": v for k, v in test_price_metrics.items()}
        )

        price_plot_path = save_actual_vs_predicted(
            test_df[DATE_COLUMN],
            test_price_true,
            test_price_pred,
            os.path.join(artifacts_dir, PRICE_PLOT_FILENAME),
            "Aktual vs Prediksi Harga Emas (USD) - Data Testing",
            "Harga Emas (USD)",
        )
        price_residual_path = save_residual_plot(
            test_df[DATE_COLUMN],
            test_price_true,
            test_price_pred,
            os.path.join(artifacts_dir, PRICE_RESIDUAL_FILENAME),
            "Residual Harga Emas (Aktual - Prediksi) pada Data Testing",
            "Residual (USD)",
        )
        artifact_paths.extend([price_plot_path, price_residual_path])

        test_predictions["actual_price"] = test_price_true
        test_predictions["predicted_price"] = test_price_pred

        summary["raw_price_path"] = raw_price_path
        summary["predictions"]["price"] = {
            "train": train_price_metrics,
            "test": test_price_metrics,
        }
        summary["train_price"] = train_price_metrics
        summary["test_price"] = test_price_metrics

        mlflow.log_metrics(logged_metrics)
        log_model(model)

        write_outputs(output_dir, run_id, summary)
        metrics_path = os.path.join(output_dir, METRICS_FILENAME)
        artifact_paths.append(metrics_path)

        test_predictions_path = os.path.join(
            artifacts_dir, TEST_PREDICTIONS_FILENAME
        )
        test_predictions.to_csv(test_predictions_path, index=False)
        artifact_paths.append(test_predictions_path)

        for artifact_path in artifact_paths:
            if not os.path.isfile(artifact_path):
                raise FileNotFoundError(
                    f"Artefak wajib tidak ditemukan: {artifact_path}"
                )
            mlflow.log_artifact(artifact_path)

        download_model_artifact(run_id, output_dir)

        print(f"\nSelesai. MLflow run id: {run_id}")


if __name__ == "__main__":
    main()
