dim_z = 100 #input dimension for samples
dim_channel = 1 #reserved for future use if multi=channels
events_num=5 #kind of events
generated_dim=32
"""Training script for multivariate conditional WGAN-GP.

Expected CSV format:
- 504 feature columns: Load(168) + Wind(168) + Solar(168)
- Last column: output_label with values {0, 1, 2}

The season label column is intentionally ignored.
"""

import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import Discriminator, Generator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_data(csv_path: str, save_scaler_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)

    if "output_label" not in df.columns:
        raise ValueError("CSV must contain column: output_label")

    # Keep only the first 504 feature columns by requirement.
    x_raw = df.iloc[:, :504].astype(np.float32).values
    y = df["output_label"].astype(int).values

    # Fit and save scaler for inference-time inverse transform.
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    x_scaled = scaler.fit_transform(x_raw).astype(np.float32)
    joblib.dump(scaler, save_scaler_path)

    return x_scaled, y


def compute_gradient_penalty(
    critic: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real_samples)

    interpolates = alpha * real_samples + (1.0 - alpha) * fake_samples
    interpolates = interpolates.requires_grad_(True)

    critic_interpolates = critic(interpolates, labels)
    grad_outputs = torch.ones_like(critic_interpolates, device=device)

    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Flatten gradient from [batch, 504] to [batch, 504] explicitly for norm.
    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp


def plot_losses(history: Dict[str, List[float]], out_path: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(history["g_loss"], label="Generator Loss")
    plt.plot(history["d_loss"], label="Critic Loss")
    plt.plot(history["wasserstein"], label="Wasserstein Distance")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Distance")
    plt.title("Training Curves (Conditional WGAN-GP)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multivariate conditional WGAN-GP")
    parser.add_argument("--data", type=str, default="shandong_gan_ready_final.csv")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--noise-dim", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.5, 0.9))
    parser.add_argument("--n-critic", type=int, default=5)
    parser.add_argument("--gp-lambda", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="outputs")
    parser.add_argument("--scaler-path", type=str, default="shandong_scaler.pkl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x, y = prepare_data(args.data, args.scaler_path)
    print(f"Training samples: {x.shape[0]}, feature_dim: {x.shape[1]}")
    unique, counts = np.unique(y, return_counts=True)
    print("Label distribution:", dict(zip(unique.tolist(), counts.tolist())))

    dataset = TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y).long(),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    generator = Generator(
        noise_dim=args.noise_dim,
        num_classes=args.num_classes,
        feature_dim=504,
    ).to(device)
    critic = Discriminator(
        num_classes=args.num_classes,
        feature_dim=504,
    ).to(device)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=tuple(args.betas))
    optimizer_d = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=tuple(args.betas))

    history: Dict[str, List[float]] = {"g_loss": [], "d_loss": [], "wasserstein": []}
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        g_losses = []
        d_losses = []
        w_dists = []

        for real_x, labels in loader:
            real_x = real_x.to(device)
            labels = labels.to(device)
            batch_size = real_x.size(0)

            for _ in range(args.n_critic):
                noise = torch.randn(batch_size, args.noise_dim, device=device)
                fake_x = generator(noise, labels).detach()

                real_score = critic(real_x, labels)
                fake_score = critic(fake_x, labels)
                wasserstein_dist = real_score.mean() - fake_score.mean()

                gp = compute_gradient_penalty(
                    critic=critic,
                    real_samples=real_x,
                    fake_samples=fake_x,
                    labels=labels,
                    device=device,
                )

                d_loss = -wasserstein_dist + args.gp_lambda * gp

                optimizer_d.zero_grad(set_to_none=True)
                d_loss.backward()
                optimizer_d.step()

            noise = torch.randn(batch_size, args.noise_dim, device=device)
            gen_x = generator(noise, labels)
            g_loss = -critic(gen_x, labels).mean()

            optimizer_g.zero_grad(set_to_none=True)
            g_loss.backward()
            optimizer_g.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            w_dists.append(wasserstein_dist.item())
            global_step += 1

        epoch_g = float(np.mean(g_losses))
        epoch_d = float(np.mean(d_losses))
        epoch_w = float(np.mean(w_dists))
        history["g_loss"].append(epoch_g)
        history["d_loss"].append(epoch_d)
        history["wasserstein"].append(epoch_w)

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch [{epoch:04d}/{args.epochs}] "
                f"G: {epoch_g:.4f} | D: {epoch_d:.4f} | W-dist: {epoch_w:.4f}"
            )

    ckpt = {
        "generator": generator.state_dict(),
        "critic": critic.state_dict(),
        "args": vars(args),
        "history": history,
    }
    ckpt_path = os.path.join(run_dir, "cgan_wgangp_504.pth")
    torch.save(ckpt, ckpt_path)

    history_path = os.path.join(run_dir, "loss_history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    loss_png = os.path.join(run_dir, "loss_curve.png")
    plot_losses(history, loss_png)

    config_path = os.path.join(run_dir, "train_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("Training completed.")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Loss csv:   {history_path}")
    print(f"Loss plot:  {loss_png}")
    print(f"Scaler:     {args.scaler_path}")


if __name__ == "__main__":
    main()
