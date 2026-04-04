"""Scenario-generation evaluation metrics and visualizations.

This script evaluates generated scenarios against real historical scenarios.
It is designed for generation quality assessment, not forecasting accuracy.
"""

import argparse
import json
import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated multivariate scenarios")
    parser.add_argument("--real-data", type=str, required=True, help="Real dataset CSV")
    parser.add_argument("--generated-data", type=str, required=True, help="Generated dataset CSV")
    parser.add_argument(
        "--label",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Optional: filter real data by output_label to compare same condition",
    )
    parser.add_argument("--out-dir", type=str, default="evaluation", help="Directory to save metrics and figures")
    parser.add_argument("--bins", type=int, default=60, help="Histogram bins")
    parser.add_argument("--max-pairs", type=int, default=100, help="Max samples for diversity metric")
    return parser.parse_args()


def _load_features_and_label(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    df = pd.read_csv(path)
    if df.shape[1] < 504:
        raise ValueError(f"{path} has only {df.shape[1]} columns, expected at least 504")

    x = df.iloc[:, :504].astype(float).values
    y = None
    if "output_label" in df.columns:
        y = df["output_label"].astype(int).values
    return x, y


def _split_blocks(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Shape transform: [N, 504] -> three [N, 168] blocks.
    load = x[:, 0:168]
    wind = x[:, 168:336]
    solar = x[:, 336:504]
    return load, wind, solar


def _flatten_blocks(x: np.ndarray) -> Dict[str, np.ndarray]:
    load, wind, solar = _split_blocks(x)
    return {
        "load": load.reshape(-1),
        "wind": wind.reshape(-1),
        "solar": solar.reshape(-1),
    }


def _w1_quantile(a: np.ndarray, b: np.ndarray, n_quantiles: int = 1000) -> float:
    q = np.linspace(0.0, 1.0, n_quantiles)
    qa = np.quantile(a, q)
    qb = np.quantile(b, q)
    return float(np.mean(np.abs(qa - qb)))


def _safe_corrcoef(m: np.ndarray) -> np.ndarray:
    if m.shape[0] < 2:
        return np.eye(m.shape[1])
    c = np.corrcoef(m, rowvar=False)
    if np.isnan(c).any():
        c = np.nan_to_num(c, nan=0.0)
        np.fill_diagonal(c, 1.0)
    return c


def _pairwise_distance_mean(x: np.ndarray, max_samples: int) -> float:
    if len(x) < 2:
        return 0.0
    if len(x) > max_samples:
        idx = np.random.choice(len(x), size=max_samples, replace=False)
        x = x[idx]

    d_sum = 0.0
    cnt = 0
    for i in range(len(x)):
        diff = x[i + 1 :] - x[i]
        d = np.sqrt(np.sum(diff * diff, axis=1))
        d_sum += float(d.sum())
        cnt += len(d)
    return d_sum / max(cnt, 1)


def _profile_metrics(real: np.ndarray, gen: np.ndarray, name: str) -> Dict[str, float]:
    real_m = real.mean(axis=0)
    gen_m = gen.mean(axis=0)
    real_s = real.std(axis=0)
    gen_s = gen.std(axis=0)

    # Ramp: delta between adjacent hours per scenario.
    real_r = np.diff(real, axis=1)
    gen_r = np.diff(gen, axis=1)

    return {
        f"{name}_mean_profile_mae": float(np.mean(np.abs(real_m - gen_m))),
        f"{name}_std_profile_mae": float(np.mean(np.abs(real_s - gen_s))),
        f"{name}_ramp_mean_abs_diff": float(np.abs(real_r.mean() - gen_r.mean())),
        f"{name}_ramp_std_abs_diff": float(np.abs(real_r.std() - gen_r.std())),
    }


def _energy_matrix(x: np.ndarray) -> np.ndarray:
    load, wind, solar = _split_blocks(x)
    # Shape: [N, 3], each column is weekly energy-like aggregate.
    return np.stack([load.sum(axis=1), wind.sum(axis=1), solar.sum(axis=1)], axis=1)


def compute_metrics(real_x: np.ndarray, gen_x: np.ndarray, max_pairs: int) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    real_blocks = _flatten_blocks(real_x)
    gen_blocks = _flatten_blocks(gen_x)

    for key in ["load", "wind", "solar"]:
        metrics[f"{key}_w1_quantile"] = _w1_quantile(real_blocks[key], gen_blocks[key])

        lo = np.percentile(real_blocks[key], 1)
        hi = np.percentile(real_blocks[key], 99)
        in_range = np.logical_and(gen_blocks[key] >= lo, gen_blocks[key] <= hi)
        metrics[f"{key}_p1_p99_coverage"] = float(in_range.mean())

    load_r, wind_r, solar_r = _split_blocks(real_x)
    load_g, wind_g, solar_g = _split_blocks(gen_x)
    metrics.update(_profile_metrics(load_r, load_g, "load"))
    metrics.update(_profile_metrics(wind_r, wind_g, "wind"))
    metrics.update(_profile_metrics(solar_r, solar_g, "solar"))

    energy_real = _energy_matrix(real_x)
    energy_gen = _energy_matrix(gen_x)

    er_mean = energy_real.mean(axis=0)
    eg_mean = energy_gen.mean(axis=0)
    er_std = energy_real.std(axis=0)
    eg_std = energy_gen.std(axis=0)

    metrics["energy_mean_l1"] = float(np.mean(np.abs(er_mean - eg_mean)))
    metrics["energy_std_l1"] = float(np.mean(np.abs(er_std - eg_std)))

    corr_real = _safe_corrcoef(energy_real)
    corr_gen = _safe_corrcoef(energy_gen)
    metrics["cross_var_corr_frobenius"] = float(np.linalg.norm(corr_real - corr_gen, ord="fro"))

    div_real = _pairwise_distance_mean(real_x, max_pairs)
    div_gen = _pairwise_distance_mean(gen_x, max_pairs)
    metrics["diversity_real_pairdist_mean"] = float(div_real)
    metrics["diversity_gen_pairdist_mean"] = float(div_gen)
    metrics["diversity_ratio_gen_over_real"] = float(div_gen / (div_real + 1e-8))

    return metrics


def plot_mean_std_profiles(real_x: np.ndarray, gen_x: np.ndarray, save_path: str) -> None:
    names = ["Load", "Wind", "Solar"]
    real_blocks = _split_blocks(real_x)
    gen_blocks = _split_blocks(gen_x)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)
    t = np.arange(168)

    for ax, name, real, gen in zip(axes, names, real_blocks, gen_blocks):
        rm, rs = real.mean(axis=0), real.std(axis=0)
        gm, gs = gen.mean(axis=0), gen.std(axis=0)

        ax.plot(t, rm, label="Real mean", linewidth=2.0)
        ax.fill_between(t, rm - rs, rm + rs, alpha=0.18)
        ax.plot(t, gm, label="Gen mean", linewidth=2.0)
        ax.fill_between(t, gm - gs, gm + gs, alpha=0.18)
        ax.set_title(name)
        ax.set_xlabel("Hour index")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Power (MW)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Scenario Profiles: Real vs Generated")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_distribution_hist(real_x: np.ndarray, gen_x: np.ndarray, bins: int, save_path: str) -> None:
    real_blocks = _flatten_blocks(real_x)
    gen_blocks = _flatten_blocks(gen_x)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, key, title in zip(axes, ["load", "wind", "solar"], ["Load", "Wind", "Solar"]):
        ax.hist(real_blocks[key], bins=bins, alpha=0.45, density=True, label="Real")
        ax.hist(gen_blocks[key], bins=bins, alpha=0.45, density=True, label="Gen")
        ax.set_title(f"{title} marginal distribution")
        ax.set_xlabel("Power (MW)")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)
        ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_correlation_heatmap(real_x: np.ndarray, gen_x: np.ndarray, save_path: str) -> None:
    labels = ["Load", "Wind", "Solar"]

    corr_real = _safe_corrcoef(_energy_matrix(real_x))
    corr_gen = _safe_corrcoef(_energy_matrix(gen_x))
    corr_diff = corr_gen - corr_real

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    mats = [corr_real, corr_gen, corr_diff]
    titles = ["Real cross-var corr", "Gen cross-var corr", "Gen - Real"]
    vmins = [-1, -1, -1]
    vmaxs = [1, 1, 1]

    for ax, mat, title, vmin, vmax in zip(axes, mats, titles, vmins, vmaxs):
        im = ax.imshow(mat, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    real_x, real_y = _load_features_and_label(args.real_data)
    gen_x, _ = _load_features_and_label(args.generated_data)

    if args.label is not None:
        if real_y is None:
            raise ValueError("--label is provided but real-data has no output_label column")
        mask = real_y == args.label
        real_x = real_x[mask]

    if len(real_x) == 0:
        raise ValueError("No real samples after filtering. Check --label and real-data.")
    if len(gen_x) == 0:
        raise ValueError("No generated samples found.")

    metrics = compute_metrics(real_x, gen_x, max_pairs=args.max_pairs)

    metrics_json = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    metrics_csv = os.path.join(args.out_dir, "metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)

    profile_png = os.path.join(args.out_dir, "mean_std_profiles.png")
    dist_png = os.path.join(args.out_dir, "marginal_distributions.png")
    corr_png = os.path.join(args.out_dir, "cross_var_correlations.png")

    plot_mean_std_profiles(real_x, gen_x, profile_png)
    plot_distribution_hist(real_x, gen_x, bins=args.bins, save_path=dist_png)
    plot_correlation_heatmap(real_x, gen_x, corr_png)

    print("Evaluation completed.")
    print(f"Metrics JSON: {metrics_json}")
    print(f"Metrics CSV:  {metrics_csv}")
    print(f"Profile fig:  {profile_png}")
    print(f"Dist fig:     {dist_png}")
    print(f"Corr fig:     {corr_png}")


if __name__ == "__main__":
    main()
