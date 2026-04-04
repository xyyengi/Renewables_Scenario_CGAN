"""Plot generated load/wind/solar curves in one figure."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one generated scenario (Load/Wind/Solar)")
    parser.add_argument("--input", type=str, required=True, help="Generated CSV path")
    parser.add_argument("--sample-index", type=int, default=0, help="Row index to visualize")
    parser.add_argument("--output", type=str, default="generated_curves.png", help="Output figure path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    if args.sample_index < 0 or args.sample_index >= len(df):
        raise IndexError(f"sample-index must be in [0, {len(df) - 1}]")

    row = df.iloc[args.sample_index]

    # Explicit slicing to keep semantic mapping clear:
    # [0:168] -> load, [168:336] -> wind, [336:504] -> solar.
    load_curve = row.iloc[0:168].astype(float).values
    wind_curve = row.iloc[168:336].astype(float).values
    solar_curve = row.iloc[336:504].astype(float).values

    hours = list(range(168))

    plt.figure(figsize=(12, 5))
    plt.plot(hours, load_curve, label="Load", linewidth=1.8)
    plt.plot(hours, wind_curve, label="Wind", linewidth=1.8)
    plt.plot(hours, solar_curve, label="Solar", linewidth=1.8)

    plt.title(f"Generated Scenario #{args.sample_index}")
    plt.xlabel("Hour Index (0-167)")
    plt.ylabel("Power (MW)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.output, dpi=220)
    plt.close()

    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
