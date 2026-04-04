"""Summarize evaluation metrics across multiple label runs."""

import argparse
import glob
import json
import os

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize metrics from eval_label*/metrics.json")
    parser.add_argument("--root", type=str, required=True, help="Root experiment directory")
    parser.add_argument("--output", type=str, default="metrics_summary.csv", help="Summary CSV output")
    return parser.parse_args()


def composite_score(row: pd.Series) -> float:
    # Smaller is better; use only distance/error-style terms.
    keys = [
        "load_w1_quantile",
        "wind_w1_quantile",
        "solar_w1_quantile",
        "load_mean_profile_mae",
        "wind_mean_profile_mae",
        "solar_mean_profile_mae",
        "cross_var_corr_frobenius",
        "energy_mean_l1",
    ]
    score = 0.0
    for k in keys:
        if k in row and pd.notna(row[k]):
            score += float(row[k])
    return score


def main() -> None:
    args = parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, "eval_label*", "metrics.json")))
    if not paths:
        raise FileNotFoundError(f"No metrics.json found under: {args.root}/eval_label*/")

    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        label_name = os.path.basename(os.path.dirname(p)).replace("eval_label", "")
        data["label"] = label_name
        data["metrics_path"] = p
        rows.append(data)

    df = pd.DataFrame(rows)
    df["composite_score_smaller_better"] = df.apply(composite_score, axis=1)
    df = df.sort_values(by="label").reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)

    show_cols = [
        "label",
        "load_w1_quantile",
        "wind_w1_quantile",
        "solar_w1_quantile",
        "cross_var_corr_frobenius",
        "diversity_ratio_gen_over_real",
        "composite_score_smaller_better",
    ]
    show_cols = [c for c in show_cols if c in df.columns]

    print("Saved summary:", args.output)
    print(df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
