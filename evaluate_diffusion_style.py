"""Evaluate CGAN outputs with diffusion-style scenario metrics.

This adapter reproduces the metric definitions used in the diffusion project and
bridges the data-shape difference between:
- Real weekly samples: [N, 504]
- CGAN generated pool: [M, 504]

It constructs scenario tensors as:
- samples: [N, n_samples, C, L]
- actual:  [N, C, L]

where C=3 channels (wind, solar, load) and L=168 hours.
"""

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


CHANNEL_NAMES = ["wind", "solar", "load"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diffusion-style metrics for CGAN generated scenarios")
    parser.add_argument("--real-data", type=str, required=True, help="Real dataset CSV (germany_data_final.csv)")
    parser.add_argument("--generated-data", type=str, required=True, help="Generated CSV (e.g., generated_label1.csv)")
    parser.add_argument("--label", type=int, default=None, choices=[0, 1, 2], help="Optional label filter on real data")
    parser.add_argument("--n-samples", type=int, default=10, help="Scenario count per real sample")
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default="with_replacement",
        choices=["with_replacement", "without_replacement", "shuffle_once"],
        help="How to draw scenarios from generated pool for each real sample",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="evaluation_diffusion_style")
    return parser.parse_args()


def _load_features_and_label(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if df.shape[1] < 504:
        raise ValueError(f"{path} has only {df.shape[1]} columns, expected >= 504")

    x = df.iloc[:, :504].astype(float).values
    if "output_label" in df.columns:
        y = df["output_label"].astype(int).values
    else:
        y = np.full((len(df),), -1, dtype=int)
    return x, y


def _to_channel_tensor(x_504: np.ndarray) -> np.ndarray:
    """Convert [K, 504] to [K, C=3, L=168] in diffusion channel order.

    Source order in CSV is [load(168), wind(168), solar(168)].
    Target channel order for diffusion evaluator is [wind, solar, load].
    """
    load = x_504[:, 0:168]
    wind = x_504[:, 168:336]
    solar = x_504[:, 336:504]
    return np.stack([wind, solar, load], axis=1)


def _build_samples_tensor(
    gen_pool: np.ndarray,
    n_real: int,
    n_samples: int,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct [N, n_samples, 504] by sampling from generated pool."""
    m = len(gen_pool)
    if m == 0:
        raise ValueError("Generated pool is empty.")

    if mode == "shuffle_once":
        if m < n_samples:
            raise ValueError("shuffle_once mode requires generated count >= n_samples")
        idx = np.arange(m)
        rng.shuffle(idx)
        chosen = gen_pool[idx[:n_samples]]
        out = np.repeat(chosen[np.newaxis, :, :], repeats=n_real, axis=0)
        return out

    out = np.zeros((n_real, n_samples, gen_pool.shape[1]), dtype=float)
    for i in range(n_real):
        if mode == "with_replacement":
            pick = rng.integers(0, m, size=n_samples)
        else:
            replace = m < n_samples
            pick = rng.choice(m, size=n_samples, replace=replace)
        out[i] = gen_pool[pick]
    return out


def compute_coverage_rate(samples: np.ndarray, actual: np.ndarray, quantile: float = 1.0) -> float:
    alpha = quantile
    lower_q = (1.0 - alpha) / 2.0
    upper_q = 1.0 - lower_q
    lower_bound = np.quantile(samples, lower_q, axis=1)
    upper_bound = np.quantile(samples, upper_q, axis=1)
    covered = (actual >= lower_bound) & (actual <= upper_bound)
    return float(np.sum(covered) / (actual.shape[0] * actual.shape[1]) * 100.0)


def compute_scenario_width(
    samples: np.ndarray,
    actual: np.ndarray,
    quantile: float = 1.0,
    global_range: float = None,
) -> float:
    alpha = quantile
    lower_q = (1.0 - alpha) / 2.0
    upper_q = 1.0 - lower_q
    lower_bound = np.quantile(samples, lower_q, axis=1)
    upper_bound = np.quantile(samples, upper_q, axis=1)
    width = upper_bound - lower_bound

    if global_range is not None and global_range > 0:
        width_normalized = width / global_range
    else:
        actual_range = np.max(actual) - np.min(actual)
        if actual_range > 0:
            width_normalized = width / actual_range
        else:
            sample_range = np.max(width)
            width_normalized = width / max(sample_range, 1e-6)
    return float(np.mean(width_normalized) * 100.0)


def compute_energy_score(samples: np.ndarray, actual: np.ndarray) -> float:
    n_samples = samples.shape[1]

    term1 = 0.0
    pair_count = 0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            diff = samples[:, i, :] - samples[:, j, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=1))
            term1 += float(np.mean(dist))
            pair_count += 1
    term1 = term1 * 2.0 / (n_samples * (n_samples - 1)) if n_samples > 1 else 0.0

    term2 = 0.0
    for i in range(n_samples):
        diff = samples[:, i, :] - actual
        dist = np.sqrt(np.sum(diff ** 2, axis=1))
        term2 += float(np.mean(dist))
    term2 = term2 * 2.0 / n_samples

    return float(term1 - term2)


def compute_crps(samples: np.ndarray, actual: np.ndarray) -> float:
    n, n_samples, l = samples.shape
    crps_values = np.zeros(n * l, dtype=float)
    idx = 0

    for i in range(n):
        for t in range(l):
            x = samples[i, :, t]
            y = actual[i, t]

            term1 = np.mean(np.abs(x - y))
            term2 = 0.0
            for a in range(n_samples):
                for b in range(a + 1, n_samples):
                    term2 += abs(x[a] - x[b])
            if n_samples > 1:
                term2 = term2 / (n_samples * (n_samples - 1) / 2.0)
            term2 = 0.5 * term2

            crps_values[idx] = term1 - term2
            idx += 1

    return float(np.mean(crps_values))


def compute_multivariate_energy_score(samples: np.ndarray, actual: np.ndarray) -> float:
    _, n_samples, c, l = samples.shape
    samples_flat = samples.reshape(samples.shape[0], n_samples, c * l)
    actual_flat = actual.reshape(actual.shape[0], c * l)

    term1 = 0.0
    for i in range(n_samples):
        diff = samples_flat[:, i, :] - actual_flat
        dist = np.sqrt(np.sum(diff ** 2, axis=1))
        term1 += float(np.mean(dist))
    term1 = term1 / n_samples

    term2 = 0.0
    pair_count = 0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            diff = samples_flat[:, i, :] - samples_flat[:, j, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=1))
            term2 += float(np.mean(dist))
            pair_count += 1
    term2 = (term2 / pair_count) if pair_count > 0 else 0.0
    term2 = 0.5 * term2

    return float(term1 - term2)


def compute_reliability(
    samples: np.ndarray,
    actual: np.ndarray,
    confidence_levels: Sequence[float],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for conf in confidence_levels:
        lower_q = (1.0 - conf) / 2.0
        upper_q = 1.0 - lower_q
        lower_bound = np.quantile(samples, lower_q, axis=1)
        upper_bound = np.quantile(samples, upper_q, axis=1)
        covered = (actual >= lower_bound) & (actual <= upper_bound)
        coverage = float(np.sum(covered) / (actual.shape[0] * actual.shape[1]) * 100.0)
        ideal = conf * 100.0
        out[f"coverage_{int(conf * 100)}%"] = coverage
        out[f"coverage_deviation_{int(conf * 100)}%"] = coverage - ideal
    return out


def _acf_for_series_batch(series_batch: np.ndarray, max_lag: int) -> np.ndarray:
    n, l = series_batch.shape
    out = np.zeros(max_lag, dtype=float)
    out[0] = 1.0
    for lag in range(1, max_lag):
        vals: List[float] = []
        for i in range(n):
            s = series_batch[i]
            if l <= lag:
                continue
            mean = np.mean(s)
            numerator = np.sum((s[:-lag] - mean) * (s[lag:] - mean))
            denominator = np.sum((s - mean) ** 2)
            if denominator > 0:
                vals.append(float(numerator / denominator))
        out[lag] = float(np.mean(vals)) if vals else 0.0
    return out


def compute_acf(samples: np.ndarray, actual: np.ndarray, max_lag: int = 24) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = samples.shape[1]
    acf_actual = _acf_for_series_batch(actual, max_lag=max_lag)
    acf_s = np.zeros((n_samples, max_lag), dtype=float)
    for s in range(n_samples):
        acf_s[s] = _acf_for_series_batch(samples[:, s, :], max_lag=max_lag)
    return acf_actual, np.mean(acf_s, axis=0), np.std(acf_s, axis=0)


def evaluate_multichannel(
    samples: np.ndarray,
    actual: np.ndarray,
    channel_names: Sequence[str],
    quantiles: Sequence[float],
) -> Dict[str, float]:
    _, _, c, _ = samples.shape
    metrics: Dict[str, float] = {}

    global_ranges = []
    for i in range(c):
        actual_c = actual[:, i, :]
        global_ranges.append(float(np.max(actual_c) - np.min(actual_c)))

    metrics["multivariate_es"] = compute_multivariate_energy_score(samples, actual)

    for i, name in enumerate(channel_names):
        samples_c = samples[:, :, i, :]
        actual_c = actual[:, i, :]

        metrics[f"{name}_crps"] = compute_crps(samples_c, actual_c)
        metrics[f"{name}_energy_score"] = compute_energy_score(samples_c, actual_c)

        for q in quantiles:
            q_name = f"{int(q * 100)}%"
            metrics[f"{name}_coverage_{q_name}"] = compute_coverage_rate(samples_c, actual_c, q)
            metrics[f"{name}_width_{q_name}"] = compute_scenario_width(
                samples_c,
                actual_c,
                q,
                global_range=global_ranges[i],
            )

        reliability = compute_reliability(samples_c, actual_c, confidence_levels=[0.80, 0.90, 0.95])
        for k, v in reliability.items():
            metrics[f"{name}_{k}"] = v

        acf_actual, acf_mean, _ = compute_acf(samples_c, actual_c, max_lag=24)
        metrics[f"{name}_acf_mae"] = float(np.mean(np.abs(acf_actual - acf_mean)))

    metrics["total_crps"] = float(np.mean([metrics[f"{name}_crps"] for name in channel_names]))
    metrics["total_energy_score"] = float(np.mean([metrics[f"{name}_energy_score"] for name in channel_names]))
    metrics["total_coverage_100%"] = float(np.mean([metrics[f"{name}_coverage_100%"] for name in channel_names]))
    metrics["total_width_100%"] = float(np.mean([metrics[f"{name}_width_100%"] for name in channel_names]))
    metrics["total_acf_mae"] = float(np.mean([metrics[f"{name}_acf_mae"] for name in channel_names]))
    return metrics


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    real_x, real_y = _load_features_and_label(args.real_data)
    gen_x, _ = _load_features_and_label(args.generated_data)

    if args.label is not None:
        if np.all(real_y == -1):
            raise ValueError("--label provided but real-data has no output_label")
        real_x = real_x[real_y == args.label]

    if len(real_x) == 0:
        raise ValueError("No real samples after filtering.")
    if len(gen_x) == 0:
        raise ValueError("No generated samples found.")

    samples_504 = _build_samples_tensor(
        gen_pool=gen_x,
        n_real=len(real_x),
        n_samples=args.n_samples,
        mode=args.sampling_mode,
        rng=rng,
    )

    actual_tensor = _to_channel_tensor(real_x)
    samples_tensor = np.stack([_to_channel_tensor(samples_504[i]) for i in range(samples_504.shape[0])], axis=0)

    metrics = evaluate_multichannel(
        samples=samples_tensor,
        actual=actual_tensor,
        channel_names=CHANNEL_NAMES,
        quantiles=[1.0, 0.9, 0.8],
    )

    meta = {
        "real_data": args.real_data,
        "generated_data": args.generated_data,
        "label": args.label,
        "n_real": int(len(real_x)),
        "generated_pool": int(len(gen_x)),
        "n_samples": int(args.n_samples),
        "sampling_mode": args.sampling_mode,
        "seed": int(args.seed),
        "channel_order": CHANNEL_NAMES,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_json = os.path.join(args.out_dir, "metrics_diffusion_style.json")
    metrics_csv = os.path.join(args.out_dir, "metrics_diffusion_style.csv")
    meta_json = os.path.join(args.out_dir, "meta_diffusion_style.json")

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Diffusion-style evaluation completed.")
    print(f"Metrics JSON: {metrics_json}")
    print(f"Metrics CSV:  {metrics_csv}")
    print(f"Meta JSON:    {meta_json}")


if __name__ == "__main__":
    main()
