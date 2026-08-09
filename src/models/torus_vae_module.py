from collections.abc import Callable

import torch
from torch.optim.lr_scheduler import LRScheduler

from src.models.components.vae import SimpleVAE
from src.models.vae_module import VAEModule
from src.utils.topology_utils import project_to_torus
from src.utils.topology_utils import torus_distance_matrix


class TorusVAEModule(VAEModule):
    """VAE whose Gaussian cover-space latent is projected onto a flat torus."""

    def __init__(
        self,
        model: SimpleVAE,
        optimizer: Callable[..., torch.optim.Optimizer],
        recon_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scheduler: Callable[..., LRScheduler] | None = None,
        prior_mean: float = 0.5,
        prior_variance: float = 0.1,
        lr: float = 1e-3,
        kl_weight: float = 1.0,
        scheduler_monitor: str = "val_loss",
    ) -> None:
        if prior_variance <= 0:
            raise ValueError("prior_variance must be positive")

        super().__init__(
            model=model,
            optimizer=optimizer,
            recon_loss=recon_loss,
            scheduler=scheduler,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            lr=lr,
            kl_weight=kl_weight,
            scheduler_monitor=scheduler_monitor,
        )
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance

    def project_latent(self, z: torch.Tensor) -> torch.Tensor:
        return project_to_torus(z, stack=True)

    def latent_distance_matrix(self, means: torch.Tensor) -> torch.Tensor:
        return torus_distance_matrix(means)

    def kl_loss(self, mu: torch.Tensor, covariance_parameters: torch.Tensor) -> torch.Tensor:
        return self.gaussian_kl_to_isotropic_prior(
            mu,
            covariance_parameters,
            prior_mean=self.prior_mean,
            prior_variance=self.prior_variance,
        )
