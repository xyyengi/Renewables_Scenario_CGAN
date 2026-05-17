"""Inference script for multivariate conditional CGAN (WGAN-GP trained).

Generates normalized 504-dim samples conditioned on output_label,
then restores physical units (MW) via inverse transform of shandong_scaler.pkl.
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
import torch

from model import Generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multivariate scenarios with trained CGAN")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--label", type=int, required=True, choices=[0, 1, 2], help="output_label condition")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of scenarios to generate")
    parser.add_argument("--noise-dim", type=int, default=None, help="Override noise dimension")
    parser.add_argument("--scaler-path", type=str, default="germany_scaler.pkl", help="Path to fitted scaler")
    parser.add_argument("--output", type=str, default="generated_samples.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_columns() -> list:
    load_cols = [f"Load_{i}" for i in range(168)]
    wind_cols = [f"Wind_{i}" for i in range(168)]
    solar_cols = [f"Solar_{i}" for i in range(168)]
    return load_cols + wind_cols + solar_cols


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.scaler_path):
        raise FileNotFoundError(
            f"Scaler file not found: {args.scaler_path}. "
            "Please train first or provide the correct --scaler-path."
        )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_args = checkpoint.get("args", {})

    noise_dim = args.noise_dim if args.noise_dim is not None else int(train_args.get("noise_dim", 128))
    num_classes = int(train_args.get("num_classes", 3))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(
        noise_dim=noise_dim,
        num_classes=num_classes,
        feature_dim=504,
    ).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    labels = torch.full((args.num_samples,), int(args.label), dtype=torch.long, device=device)
    noise = torch.randn(args.num_samples, noise_dim, device=device)

    with torch.no_grad():
        # Output shape is [num_samples, 504], in normalized range [0, 1].
        generated_norm = generator(noise, labels).cpu().numpy()

    scaler = joblib.load(args.scaler_path)
    generated_mw = scaler.inverse_transform(generated_norm)

    df = pd.DataFrame(generated_mw, columns=build_columns())
    df["output_label"] = int(args.label)
    df.to_csv(args.output, index=False)

    print("Generation completed.")
    print(f"Condition label: {args.label}")
    print(f"Samples:         {args.num_samples}")
    print(f"Output CSV:      {args.output}")


if __name__ == "__main__":
    main()
