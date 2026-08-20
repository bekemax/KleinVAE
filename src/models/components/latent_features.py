"""Fixed lifts making the decoder a well-defined function on the latent manifold."""

from __future__ import annotations

import math

import torch
from torch import nn

from src.utils.topology_utils import KLEIN_U_WIDTH, TORUS_U_PERIOD, TORUS_V_PERIOD


class LatentFeatureMap(nn.Module):
    """Base class for parameter-free lifts of a 2-d latent into the decoder's input space."""

    output_dim: int

    def forward(self, latent: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class TorusFourierFeatures(LatentFeatureMap):
    """Lift (u, v) to the flat torus in R^4; invariant under both period shifts."""

    output_dim = 4

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        theta = 2 * math.pi * latent[..., 0] / TORUS_U_PERIOD
        phi = 2 * math.pi * latent[..., 1] / TORUS_V_PERIOD
        return torch.stack([theta.cos(), theta.sin(), phi.cos(), phi.sin()], dim=-1)


class KleinFourierFeatures(LatentFeatureMap):
    """Lift (u, v) to the Klein bottle in R^5.

    With theta = pi*u and phi = 2*pi*v the deck generators are (theta, phi + 2*pi) and
    (theta + pi, -phi), so an invariant real mode pairs an even theta-frequency with cos(phi)
    and an odd one with sin(phi). These are the five lowest such modes.
    """

    output_dim = 5

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        theta = math.pi * latent[..., 0] / KLEIN_U_WIDTH
        phi = 2 * math.pi * latent[..., 1] / TORUS_V_PERIOD
        sin_phi = phi.sin()
        features = [(2 * theta).cos(), (2 * theta).sin(), theta.cos() * sin_phi, theta.sin() * sin_phi, phi.cos()]
        return torch.stack(features, dim=-1)


class RandomFourierFeatures(LatentFeatureMap):
    """Seam-free lift with no identification, so the Euclidean baseline gets a matched basis."""

    def __init__(self, latent_dim: int = 2, output_dim: int = 5, bandwidth: float = 1.0, seed: int = 0) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        self.output_dim = output_dim
        generator = torch.Generator().manual_seed(seed)
        # Buffers, not parameters: the basis is fixed at construction and restored with the checkpoint.
        self.register_buffer("frequencies", torch.randn(latent_dim, output_dim, generator=generator) * bandwidth)
        self.register_buffer("phases", torch.rand(output_dim, generator=generator) * 2 * math.pi)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.cos(latent @ self.frequencies + self.phases)
