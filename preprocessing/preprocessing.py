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
        raise FileNotFoundError(f"File dataset tidak ditemukan: {path}")

    df = pd.read_csv(path)

    required_columns = {"Date", "Price"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Kolom wajib {sorted(missing_columns)} tidak ditemukan pada {path}. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    df = df[["Date", "Price"]].rename(columns={"Price": price_column_name})

    invalid_date_mask = pd.to_datetime(
        df["Date"], format=DATE_FORMAT, errors="coerce"
    ).isna()
    if invalid_date_mask.any():
        print(
            f"[INFO] {path.name}: {int(invalid_date_mask.sum())} baris tanggal "
            "tidak dapat dikonversi dan akan dihapus."
        )

    numeric_price = pd.to_numeric(
        df[price_column_name].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )
    invalid_price_mask = numeric_price.isna()
    if invalid_price_mask.any():
        print(
            f"[INFO] {path.name}: {int(invalid_price_mask.sum())} baris harga "
            "tidak dapat dikonversi dan akan dihapus."
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
        print(f"[INFO] {label}: {duplicate_dates} duplikasi tanggal dihapus (keep first).")
        df = df.drop_duplicates(subset="Date", keep="first")

    df = df.sort_values("Date").reset_index(drop=True)

    print(
        f"[INFO] {label}: {rows_before} baris mentah -> "
        f"{rows_after_missing} setelah dropna, "
        f"{rows_outside_period} di luar periode, "
        f"{rows_weekend} baris weekend dihapus, "
        f"{len(df)} baris bersih."
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

    # Tahap 1-3: pemilihan kolom, konversi tipe data, pembersihan dasar.
    gold_raw = load_raw_dataset(gold_path, "gold_price")
    chf_raw = load_raw_dataset(chf_path, "chf_price")

    print(f"[INFO] Dataset emas mentah : {gold_raw.shape}")
    print(f"[INFO] Dataset CHF mentah  : {chf_raw.shape}")

    gold_clean, _ = _prepare_single(gold_raw, "gold_price", "Emas")
    chf_clean, _ = _prepare_single(chf_raw, "chf_price", "CHF")

    # Tahap 5: inner join pada kolom tanggal.
    merged = pd.merge(gold_clean, chf_clean, on="Date", how="inner")
    merged = merged.sort_values("Date").reset_index(drop=True)
    print(f"[INFO] Hasil inner join    : {merged.shape} (kolom: {list(merged.columns)})")

    # Tahap 6: perhitungan simple return.
    merged["return_gold"] = merged["gold_price"].pct_change(fill_method=None)
    merged["return_chf"] = merged["chf_price"].pct_change(fill_method=None)

    # Tahap 7: pembuatan fitur lag 1-10.
    for lag in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        merged[f"return_gold_lag_{lag}"] = merged["return_gold"].shift(lag)
        merged[f"return_chf_lag_{lag}"] = merged["return_chf"].shift(lag)

    # Tahap 8: hapus baris dengan return/lag tidak lengkap (tanpa imputasi).
    rows_before_final_drop = len(merged)
    final_df = merged.dropna(subset=FINAL_COLUMNS).reset_index(drop=True)
    rows_dropped = rows_before_final_drop - len(final_df)
    print(
        f"[INFO] Baris dihapus karena return/lag tidak lengkap: {rows_dropped} "
        f"(dataset final: {len(final_df)} baris)."
    )

    # Tahap 9: struktur final (Date + return + 20 fitur lag) dengan urutan benar.
    final_df = final_df[FINAL_COLUMNS]

    # Tahap 10: penyimpanan.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"[INFO] Dataset final disimpan di: {output_path}")
    print(f"[INFO] Dimensi dataset final: {final_df.shape}")

    return final_df


def main():
    final_df = preprocess_data()
    print("\n=== Ringkasan Dataset Final ===")
    print(final_df.head())
    print("...")
    print(final_df.tail())


if __name__ == "__main__":
    main()
