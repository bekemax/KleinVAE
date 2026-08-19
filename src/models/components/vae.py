from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn

CovarianceType = Literal["isotropic", "diagonal", "full"]

# Bound on the encoder outputs that are exponentiated to form the posterior
# scale. ``torch.exp`` overflows float32 near an argument of 88, and an
# unbounded encoder can reach that: run I2 of the BCE-fix sweep produced a
# non-finite scale_tril at epoch 27 even with BCEWithLogitsLoss in place, which
# then propagated NaN through the whole model.
#
# A posterior standard deviation outside ``exp(±LOG_SCALE_BOUND)`` is already
# degenerate for a latent that lives in the unit square -- exp(-10) is far below
# the pixel quantisation of the data and exp(10) spans the domain many times
# over -- so clamping here removes the overflow path without excluding any
# solution the model would plausibly converge to.
LOG_SCALE_BOUND = 10.0


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
        decoder_layers.append(nn.Linear(previous_dim, input_dim))
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

        # The isotropic and diagonal branches parameterise log-variance, so the
        # bound on log-standard-deviation doubles; the full branch's diagonal
        # entries are log-standard-deviations already.
        if self.covariance_type == "isotropic":
            log_variance = covariance_parameters[..., :1].clamp(-2 * LOG_SCALE_BOUND, 2 * LOG_SCALE_BOUND)
            scale = torch.exp(0.5 * log_variance).expand(*log_variance.shape[:-1], self.latent_dim)
            return torch.diag_embed(scale)

        if self.covariance_type == "diagonal":
            log_variance = covariance_parameters.clamp(-2 * LOG_SCALE_BOUND, 2 * LOG_SCALE_BOUND)
            return torch.diag_embed(torch.exp(0.5 * log_variance))

        values = covariance_parameters.clone()
        diagonal_mask = self._tril_rows == self._tril_cols
        values[..., diagonal_mask] = torch.exp(
            values[..., diagonal_mask].clamp(-LOG_SCALE_BOUND, LOG_SCALE_BOUND)
        )
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
        """Return Bernoulli *logits*, not probabilities.

        The decoder deliberately omits a final sigmoid so the reconstruction
        term can use ``BCEWithLogitsLoss``, which fuses the sigmoid into the
        loss via log-sum-exp.  Applying a sigmoid here and scoring with plain
        ``BCELoss`` is numerically unstable: the gradient of BCE with respect
        to a probability ``p`` is ``(p - y) / (p * (1 - p))``, which diverges
        once the sigmoid saturates to exactly 0 or 1 in float32.  Use
        :meth:`decode_probabilities` wherever an image is needed.
        """

        return self.decoder(z)

    def decode_probabilities(self, z: torch.Tensor) -> torch.Tensor:
        """Return per-pixel Bernoulli means in ``[0, 1]`` for visualization."""

        return torch.sigmoid(self.decode(z))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, covariance_parameters = self.encode(x)
        z = self.reparameterize(mu, covariance_parameters)
        reconstruction = self.decode(z)
        return reconstruction, mu, covariance_parameters
