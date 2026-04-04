"""Model definitions for multivariate conditional WGAN-GP.

Data layout per sample:
- First 168 values: load
- Next 168 values: wind
- Last 168 values: solar

Total feature dimension is 504.
"""

from typing import Tuple

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Conditional generator for 504-dim normalized scenarios."""

    def __init__(
        self,
        noise_dim: int = 128,
        num_classes: int = 3,
        label_emb_dim: int = 16,
        feature_dim: int = 504,
        hidden_dims: Tuple[int, int, int] = (512, 1024, 1024),
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.feature_dim = feature_dim
        self.label_embedding = nn.Embedding(num_classes, label_emb_dim)

        input_dim = noise_dim + label_emb_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dims[2], feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_vec = self.label_embedding(labels)
        x = torch.cat([noise, label_vec], dim=1)
        out = self.net(x)
        # Keep tensor in shape [batch, 504].
        return out


class Discriminator(nn.Module):
    """Conditional critic for WGAN-GP; outputs Wasserstein score (no sigmoid)."""

    def __init__(
        self,
        num_classes: int = 3,
        label_emb_dim: int = 16,
        feature_dim: int = 504,
        hidden_dims: Tuple[int, int, int] = (1024, 512, 256),
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.label_embedding = nn.Embedding(num_classes, label_emb_dim)

        input_dim = feature_dim + label_emb_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.15),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.15),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dims[2], 1),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_vec = self.label_embedding(labels)
        critic_input = torch.cat([x, label_vec], dim=1)
        score = self.net(critic_input)
        return score


