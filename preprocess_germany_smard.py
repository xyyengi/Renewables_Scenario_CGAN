"""Preprocess German SMARD data into weekly GAN-ready scenarios.

The script:
- reads the generation and consumption CSV files with ';' separators,
- extracts Load, Wind (offshore + onshore), and Solar,
- aligns timestamps and fills missing values,
- scales all variables to [0, 1],
- slices the series into non-overlapping 168-hour weeks,
- assigns output_label based on weekly wind+solar average,
- writes germany_data_final.csv and germany_scaler.pkl.
"""

from __future__ import annotations

import argparse
import re
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


DEFAULT_GENERATION_FILE = "Actual_generation_202301010000_202603011700_Hour (1).csv"
DEFAULT_CONSUMPTION_FILE = "Actual_consumption_202301010000_202603011700_Hour (2).csv"
DEFAULT_OUTPUT_CSV = "germany_data_final.csv"
DEFAULT_SCALER_PATH = "germany_scaler.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess German SMARD data")
    parser.add_argument("--generation-file", type=str, default=DEFAULT_GENERATION_FILE)
    parser.add_argument("--consumption-file", type=str, default=DEFAULT_CONSUMPTION_FILE)
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--scaler-path", type=str, default=DEFAULT_SCALER_PATH)
    parser.add_argument(
        "--label-method",
        type=str,
        choices=["quantile", "fixed"],
        default="quantile",
        help="Labeling strategy for weekly renewable output",
    )
    parser.add_argument(
        "--fixed-thresholds",
        type=float,
        nargs=2,
        default=(0.33, 0.66),
        metavar=("LOW", "HIGH"),
        help="Thresholds used when --label-method fixed",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs=2,
        default=(0.33, 0.66),
        metavar=("Q1", "Q2"),
        help="Quantiles used when --label-method quantile",
    )
    return parser.parse_args()


def _normalize_numeric_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().replace(" ", "")
    if text == "":
        return ""

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            return text.replace(".", "").replace(",", ".")
        return text.replace(",", "")

    if "," in text:
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+", text):
            return text.replace(",", "")
        return text.replace(",", ".")

    return text


def _to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.map(_normalize_numeric_text), errors="coerce")


def _read_smard_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False)
    df.columns = [col.strip() for col in df.columns]
    return df


def _extract_time_window(df: pd.DataFrame) -> pd.DataFrame:
    if "Start date" not in df.columns or "End date" not in df.columns:
        raise ValueError("SMARD CSV must contain 'Start date' and 'End date' columns")

    df = df.copy()
    df["Start date"] = pd.to_datetime(
        df["Start date"], format="%b %d, %Y %I:%M %p", errors="coerce"
    )
    df["End date"] = pd.to_datetime(
        df["End date"], format="%b %d, %Y %I:%M %p", errors="coerce"
    )
    df = df.dropna(subset=["Start date", "End date"])
    df = df.drop_duplicates(subset=["Start date", "End date"], keep="first")
    df = df.sort_values(["Start date", "End date"]).reset_index(drop=True)
    return df


def _prepare_merged_frame(generation_path: str, consumption_path: str) -> pd.DataFrame:
    gen = _extract_time_window(_read_smard_csv(generation_path))
    con = _extract_time_window(_read_smard_csv(consumption_path))

    gen_cols = {
        "Start date",
        "End date",
        "Wind offshore [MWh] Calculated resolutions",
        "Wind onshore [MWh] Calculated resolutions",
        "Photovoltaics [MWh] Calculated resolutions",
    }
    con_cols = {
        "Start date",
        "End date",
        "grid load [MWh] Calculated resolutions",
    }

    missing_gen = gen_cols.difference(gen.columns)
    missing_con = con_cols.difference(con.columns)
    if missing_gen:
        raise ValueError(f"Generation file missing columns: {sorted(missing_gen)}")
    if missing_con:
        raise ValueError(f"Consumption file missing columns: {sorted(missing_con)}")

    gen = gen[[
        "Start date",
        "End date",
        "Wind offshore [MWh] Calculated resolutions",
        "Wind onshore [MWh] Calculated resolutions",
        "Photovoltaics [MWh] Calculated resolutions",
    ]].copy()
    con = con[[
        "Start date",
        "End date",
        "grid load [MWh] Calculated resolutions",
    ]].copy()

    for column in [
        "Wind offshore [MWh] Calculated resolutions",
        "Wind onshore [MWh] Calculated resolutions",
        "Photovoltaics [MWh] Calculated resolutions",
    ]:
        gen[column] = _to_numeric_series(gen[column])

    con["grid load [MWh] Calculated resolutions"] = _to_numeric_series(
        con["grid load [MWh] Calculated resolutions"]
    )

    merged = pd.merge(gen, con, on=["Start date", "End date"], how="outer", sort=True)
    merged = merged.sort_values(["Start date", "End date"]).reset_index(drop=True)

    merged["Load"] = merged["grid load [MWh] Calculated resolutions"]
    merged["Wind"] = (
        merged["Wind offshore [MWh] Calculated resolutions"].fillna(0.0)
        + merged["Wind onshore [MWh] Calculated resolutions"].fillna(0.0)
    )
    merged["Solar"] = merged["Photovoltaics [MWh] Calculated resolutions"]

    merged = merged[["Start date", "End date", "Load", "Wind", "Solar"]]
    merged["Load"] = merged["Load"].interpolate(limit_direction="both")
    merged["Wind"] = merged["Wind"].interpolate(limit_direction="both")
    merged["Solar"] = merged["Solar"].interpolate(limit_direction="both")
    merged = merged.ffill().bfill()

    if merged[["Load", "Wind", "Solar"]].isna().any().any():
        raise ValueError("Unable to fill all missing values after interpolation")

    return merged


def _compute_week_labels(
    weekly_renewable_means: np.ndarray,
    method: str,
    thresholds: Tuple[float, float],
    quantiles: Tuple[float, float],
) -> np.ndarray:
    if method == "fixed":
        low, high = thresholds
    else:
        low = float(np.quantile(weekly_renewable_means, quantiles[0]))
        high = float(np.quantile(weekly_renewable_means, quantiles[1]))

    if low > high:
        raise ValueError("Lower threshold is greater than upper threshold")

    labels = np.zeros_like(weekly_renewable_means, dtype=int)
    labels[weekly_renewable_means >= high] = 2
    mid_mask = (weekly_renewable_means >= low) & (weekly_renewable_means < high)
    labels[mid_mask] = 1
    return labels


def _make_feature_columns() -> list[str]:
    return [f"Load_{i}" for i in range(168)] + [f"Wind_{i}" for i in range(168)] + [f"Solar_{i}" for i in range(168)]


def _slice_weeks(
    df_scaled: pd.DataFrame,
    label_method: str,
    fixed_thresholds: Tuple[float, float],
    quantiles: Tuple[float, float],
) -> pd.DataFrame:
    hours_per_week = 168
    num_weeks = len(df_scaled) // hours_per_week
    if num_weeks == 0:
        raise ValueError("Not enough hourly data for a single 168-hour week")

    trimmed = df_scaled.iloc[: num_weeks * hours_per_week].reset_index(drop=True)
    weekly_renewable_means = []
    weekly_rows = []

    for week_index in range(num_weeks):
        week_chunk = trimmed.iloc[week_index * hours_per_week : (week_index + 1) * hours_per_week]
        weekly_renewable_means.append(float((week_chunk["Wind"] + week_chunk["Solar"]).mean()))
        flat_features = np.concatenate([
            week_chunk["Load"].to_numpy(dtype=np.float32),
            week_chunk["Wind"].to_numpy(dtype=np.float32),
            week_chunk["Solar"].to_numpy(dtype=np.float32),
        ])
        weekly_rows.append(flat_features)

    weekly_renewable_means_arr = np.asarray(weekly_renewable_means, dtype=np.float64)
    labels = _compute_week_labels(
        weekly_renewable_means=weekly_renewable_means_arr,
        method=label_method,
        thresholds=fixed_thresholds,
        quantiles=quantiles,
    )

    columns = _make_feature_columns() + ["output_label"]
    features_df = pd.DataFrame(np.asarray(weekly_rows, dtype=np.float32), columns=_make_feature_columns())
    features_df["output_label"] = labels.astype(np.int64)
    return features_df[columns]


def main() -> None:
    args = parse_args()

    merged = _prepare_merged_frame(args.generation_file, args.consumption_file)

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    merged[["Load", "Wind", "Solar"]] = scaler.fit_transform(merged[["Load", "Wind", "Solar"]])
    joblib.dump(scaler, args.scaler_path)

    final_df = _slice_weeks(
        df_scaled=merged,
        label_method=args.label_method,
        fixed_thresholds=tuple(args.fixed_thresholds),
        quantiles=tuple(args.quantiles),
    )
    final_df.to_csv(args.output_csv, index=False)

    label_counts = final_df["output_label"].value_counts().sort_index().to_dict()
    print(f"Saved: {args.output_csv}")
    print(f"Saved: {args.scaler_path}")
    print(f"Weeks: {len(final_df)}")
    print(f"Label distribution: {label_counts}")


if __name__ == "__main__":
    main()