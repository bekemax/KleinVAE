from collections.abc import Callable
from typing import Any

import lightning as pl
import torch
from torch.optim.lr_scheduler import LRScheduler

from src.models.components.vae import SimpleVAE
from src.utils.topology_utils import euclidean_distance_matrix


class VAEModule(pl.LightningModule):
    """Standard VAE with shared training and evaluation instrumentation."""

    def __init__(
        self,
        model: SimpleVAE,
        optimizer: Callable[..., torch.optim.Optimizer],
        recon_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scheduler: Callable[..., LRScheduler] | None = None,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        lr: float = 1e-3,
        kl_weight: float = 1.0,
        scheduler_monitor: str = "val_loss",
    ) -> None:
        if prior_variance <= 0:
            raise ValueError("prior_variance must be positive")
        super().__init__()
        self.save_hyperparameters(ignore=["model", "recon_loss"])
        self.model = model
        self.recon_loss = recon_loss
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.lr = lr
        self.kl_weight = kl_weight

    def refreeze_encoder(self, trainable: bool = False) -> None:
        """Set whether encoder parameters require gradients."""

        for parameter in self.model.encoder.parameters():
            parameter.requires_grad = trainable

    def project_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Map a cover-space latent into the decoder's latent domain."""

        return z

    def latent_distance_matrix(self, means: torch.Tensor) -> torch.Tensor:
        """Pairwise distance under the model's native latent metric."""

        return euclidean_distance_matrix(means)

    def reconstruct(self, x: torch.Tensor, *, sample: bool = False) -> torch.Tensor:
        """Reconstruct inputs, deterministically from the posterior mean by default."""

        mu, covariance_parameters = self.model.encode(x)
        z = self.model.reparameterize(mu, covariance_parameters) if sample else mu
        return self.model.decode(self.project_latent(z))

    def reconstruction_loss(self, x: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        """Return the configured reconstruction term used for optimization.

        With ``BCELoss(reduction="sum")`` this is the factorized Bernoulli
        negative log-likelihood summed over pixels and averaged over the batch.
        With ``reduction="mean"`` it reproduces the original notebook's
        pixel-mean objective.
        """

        target = x if x.shape == reconstruction.shape else x.reshape(x.shape[0], -1)
        loss = self.recon_loss(reconstruction, target)
        if getattr(self.recon_loss, "reduction", None) == "sum":
            loss = loss / x.shape[0]
        return loss

    def reconstruction_nll(self, x: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        """Return per-observation reconstruction NLL for comparable logging."""

        loss = self.reconstruction_loss(x, reconstruction)
        if getattr(self.recon_loss, "reduction", None) == "mean":
            loss = loss * x[0].numel()
        return loss

    def gaussian_kl_to_isotropic_prior(
        self,
        mu: torch.Tensor,
        covariance_parameters: torch.Tensor,
        *,
        prior_mean: float,
        prior_variance: float,
    ) -> torch.Tensor:
        """Compute KL(N(mu, Sigma) || N(prior_mean, prior_variance I))."""

        if prior_variance <= 0:
            raise ValueError("prior_variance must be positive")

        scale_tril = self.model.posterior_scale_tril(covariance_parameters)
        trace = scale_tril.square().sum(dim=(-2, -1))
        centered_mean_norm = (mu - prior_mean).square().sum(dim=-1)
        posterior_log_det = 2 * torch.log(torch.diagonal(scale_tril, dim1=-2, dim2=-1)).sum(dim=-1)
        latent_dim = mu.shape[-1]
        prior_log_det = latent_dim * mu.new_tensor(prior_variance).log()
        kl_per_observation = 0.5 * ((trace + centered_mean_norm) / prior_variance - latent_dim + prior_log_det - posterior_log_det)
        return kl_per_observation.mean()

    def kl_loss(self, mu: torch.Tensor, covariance_parameters: torch.Tensor) -> torch.Tensor:
        return self.gaussian_kl_to_isotropic_prior(
            mu,
            covariance_parameters,
            prior_mean=self.prior_mean,
            prior_variance=self.prior_variance,
        )

    def _vae_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        covariance_parameters: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return training objective, configured reconstruction term, and KL."""

        reconstruction_term = self.reconstruction_loss(x, reconstruction)
        kl = self.kl_loss(mu, covariance_parameters)
        objective = reconstruction_term + self.kl_weight * kl
        return objective, reconstruction_term, kl

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, covariance_parameters = self.model.encode(x)
        z = self.model.reparameterize(mu, covariance_parameters)
        reconstruction = self.model.decode(self.project_latent(z))
        return reconstruction, mu, covariance_parameters

    def _shared_step(self, stage: str, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        x = batch[0].float()
        reconstruction, mu, covariance_parameters = self(x)
        objective, reconstruction_term, kl = self._vae_loss(x, reconstruction, mu, covariance_parameters)
        reconstruction_nll = self.reconstruction_nll(x, reconstruction)
        negative_elbo = reconstruction_nll + kl
        posterior_log_variance = self.model.posterior_log_variance(covariance_parameters).mean()

        metrics = {
            f"{stage}_loss": objective,
            f"{stage}_recon_loss": reconstruction_term,
            f"{stage}_recon_nll": reconstruction_nll,
            f"{stage}_kl_loss": kl,
            f"{stage}_negative_elbo": negative_elbo,
            f"{stage}_posterior_log_variance": posterior_log_variance,
        }
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=stage != "test",
            batch_size=x.shape[0],
            sync_dist=True,
        )
        return objective

    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        return self._shared_step("train", batch)

    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        return self._shared_step("val", batch)

    def test_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        return self._shared_step("test", batch)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())  # type: ignore[attr-defined]
        scheduler_config = self.hparams.get("scheduler")
        if scheduler_config is not None and scheduler_config != {}:
            scheduler: LRScheduler = scheduler_config(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.scheduler_monitor,
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": True,
                    "name": "learning_rate",
                },
            }
        return {"optimizer": optimizer}
