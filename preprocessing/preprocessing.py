from pathlib import Path

import pandas as pd

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

RAW_DIR = REPO_ROOT / "data" / "raw"
GOLD_PATH = RAW_DIR / "XAU_USD Historical Data.csv"
CHF_PATH = RAW_DIR / "CHF_USD Historical Data.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "harga_emas_dan_swiss_franc_preprocessing.csv"

START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
DATE_FORMAT = "%m/%d/%Y"

FINAL_COLUMNS = ["Date", "return_gold"] + [
    f"return_gold_lag_{lag}" for lag in range(1, 11)
] + [
    f"return_chf_lag_{lag}" for lag in range(1, 11)
]


def load_raw_dataset(path, price_column_name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)

    required_columns = {"Date", "Price"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Required columns {sorted(missing_columns)} not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[["Date", "Price"]].rename(columns={"Price": price_column_name})

    invalid_date_mask = pd.to_datetime(
        df["Date"], format=DATE_FORMAT, errors="coerce"
    ).isna()
    if invalid_date_mask.any():
        print(
            f"[INFO] {path.name}: {int(invalid_date_mask.sum())} date rows "
            "could not be converted and will be dropped."
        )

    numeric_price = pd.to_numeric(
        df[price_column_name].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )
    invalid_price_mask = numeric_price.isna()
    if invalid_price_mask.any():
        print(
            f"[INFO] {path.name}: {int(invalid_price_mask.sum())} price rows "
            "could not be converted and will be dropped."
        )

    df["Date"] = pd.to_datetime(df["Date"], format=DATE_FORMAT, errors="coerce")
    df[price_column_name] = numeric_price

    return df


def _prepare_single(df, price_column_name, label):
    rows_before = len(df)

    df = df.dropna(subset=["Date", price_column_name]).copy()
    rows_after_missing = len(df)

    df = df.sort_values("Date").reset_index(drop=True)

    mask_period = (df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)
    rows_outside_period = int((~mask_period).sum())
    df = df.loc[mask_period].copy()

    weekend_mask = df["Date"].dt.dayofweek >= 5
    rows_weekend = int(weekend_mask.sum())
    df = df.loc[~weekend_mask].copy()

    duplicate_dates = int(df["Date"].duplicated().sum())
    if duplicate_dates:
        print(f"[INFO] {label}: {duplicate_dates} duplicate dates dropped (keep first).")
        df = df.drop_duplicates(subset="Date", keep="first")

    df = df.sort_values("Date").reset_index(drop=True)

    print(
        f"[INFO] {label}: {rows_before} raw rows -> "
        f"{rows_after_missing} after dropna, "
        f"{rows_outside_period} outside period, "
        f"{rows_weekend} weekend rows dropped, "
        f"{len(df)} clean rows."
    )
    return df, {
        "rows_before": rows_before,
        "rows_after_missing": rows_after_missing,
        "rows_outside_period": rows_outside_period,
        "rows_weekend": rows_weekend,
        "rows_clean": len(df),
    }


# Preprocessing
def preprocess_data(gold_path=GOLD_PATH, chf_path=CHF_PATH, output_path=OUTPUT_PATH):
    gold_path = Path(gold_path)
    chf_path = Path(chf_path)
    output_path = Path(output_path)

    # Steps 1-3: column selection, data type conversion, basic cleaning.
    gold_raw = load_raw_dataset(gold_path, "gold_price")
    chf_raw = load_raw_dataset(chf_path, "chf_price")

    print(f"[INFO] Raw gold dataset : {gold_raw.shape}")
    print(f"[INFO] Raw CHF dataset  : {chf_raw.shape}")

    gold_clean, _ = _prepare_single(gold_raw, "gold_price", "Gold")
    chf_clean, _ = _prepare_single(chf_raw, "chf_price", "CHF")

    # Step 5: inner join on the date column.
    merged = pd.merge(gold_clean, chf_clean, on="Date", how="inner")
    merged = merged.sort_values("Date").reset_index(drop=True)
    print(f"[INFO] Inner join result  : {merged.shape} (columns: {list(merged.columns)})")

    # Step 6: simple return computation.
    merged["return_gold"] = merged["gold_price"].pct_change(fill_method=None)
    merged["return_chf"] = merged["chf_price"].pct_change(fill_method=None)

    # Step 7: create lag features 1-10.
    for lag in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        merged[f"return_gold_lag_{lag}"] = merged["return_gold"].shift(lag)
        merged[f"return_chf_lag_{lag}"] = merged["return_chf"].shift(lag)

    # Step 8: drop rows with incomplete returns/lags (no imputation).
    rows_before_final_drop = len(merged)
    final_df = merged.dropna(subset=FINAL_COLUMNS).reset_index(drop=True)
    rows_dropped = rows_before_final_drop - len(final_df)
    print(
        f"[INFO] Rows dropped due to incomplete returns/lags: {rows_dropped} "
        f"(final dataset: {len(final_df)} rows)."
    )

    # Step 9: final structure (Date + returns + 20 lag features) in the correct order.
    final_df = final_df[FINAL_COLUMNS]

    # Step 10: persist the dataset.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"[INFO] Final dataset saved to: {output_path}")
    print(f"[INFO] Final dataset shape: {final_df.shape}")

    return final_df


def main():
    final_df = preprocess_data()
    print("\n=== Final Dataset Summary ===")
    print(final_df.head())
    print("...")
    print(final_df.tail())


if __name__ == "__main__":
    main()
