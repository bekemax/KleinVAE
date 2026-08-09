from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn

CovarianceType = Literal["isotropic", "diagonal", "full"]


class SimpleVAE(nn.Module):
    """Fully connected VAE used by the Euclidean, torus, and Klein models."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] | None = None,
        latent_dim: int = 2,
        covariance_type: CovarianceType = "full",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if covariance_type not in {"isotropic", "diagonal", "full"}:
            raise ValueError(f"Unsupported covariance_type: {covariance_type}")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = list(hidden_dims) if hidden_dims is not None else [16, 32, 64]
        self.covariance_type = covariance_type

        if covariance_type == "isotropic":
            self.covariance_parameter_dim = 1
        elif covariance_type == "diagonal":
            self.covariance_parameter_dim = latent_dim
        else:
            self.covariance_parameter_dim = latent_dim * (latent_dim + 1) // 2

        encoder_layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend((nn.Linear(previous_dim, hidden_dim), nn.LeakyReLU()))
            previous_dim = hidden_dim
        encoder_layers.append(nn.Linear(previous_dim, latent_dim + self.covariance_parameter_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        previous_dim = latent_dim
        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.extend((nn.Linear(previous_dim, hidden_dim), nn.LeakyReLU()))
            previous_dim = hidden_dim
        decoder_layers.extend((nn.Linear(previous_dim, input_dim), nn.Sigmoid()))
        self.decoder = nn.Sequential(*decoder_layers)

        tril_rows, tril_cols = torch.tril_indices(latent_dim, latent_dim)
        self.register_buffer("_tril_rows", tril_rows, persistent=False)
        self.register_buffer("_tril_cols", tril_cols, persistent=False)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the posterior mean and unconstrained covariance parameters."""

        encoded = self.encoder(x.reshape(x.shape[0], -1))
        mu = encoded[..., : self.latent_dim]
        covariance_parameters = encoded[..., self.latent_dim :]
        return mu, covariance_parameters

    def posterior_scale_tril(self, covariance_parameters: torch.Tensor) -> torch.Tensor:
        """Construct the posterior's lower-triangular scale matrix."""

        if covariance_parameters.shape[-1] != self.covariance_parameter_dim:
            raise ValueError(
                f"Expected covariance parameter dimension {self.covariance_parameter_dim}, got {covariance_parameters.shape[-1]}"
            )

        if self.covariance_type == "isotropic":
            log_variance = covariance_parameters[..., :1]
            scale = torch.exp(0.5 * log_variance).expand(*log_variance.shape[:-1], self.latent_dim)
            return torch.diag_embed(scale)

        if self.covariance_type == "diagonal":
            return torch.diag_embed(torch.exp(0.5 * covariance_parameters))

        values = covariance_parameters.clone()
        diagonal_mask = self._tril_rows == self._tril_cols
        values[..., diagonal_mask] = torch.exp(values[..., diagonal_mask])
        scale_tril = covariance_parameters.new_zeros(*covariance_parameters.shape[:-1], self.latent_dim, self.latent_dim)
        scale_tril[..., self._tril_rows, self._tril_cols] = values
        return scale_tril

    def posterior_log_variance(self, covariance_parameters: torch.Tensor) -> torch.Tensor:
        """Return ``log(diag(Sigma))`` for metric logging and diagnostics."""

        scale_tril = self.posterior_scale_tril(covariance_parameters)
        marginal_variance = scale_tril.square().sum(dim=-1)
        return torch.log(marginal_variance.clamp_min(torch.finfo(marginal_variance.dtype).tiny))

    def reparameterize(self, mu: torch.Tensor, covariance_parameters: torch.Tensor) -> torch.Tensor:
        scale_tril = self.posterior_scale_tril(covariance_parameters)
        noise = torch.randn_like(mu)
        return mu + torch.matmul(scale_tril, noise.unsqueeze(-1)).squeeze(-1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, covariance_parameters = self.encode(x)
        z = self.reparameterize(mu, covariance_parameters)
        reconstruction = self.decode(z)
        return reconstruction, mu, covariance_parameters
