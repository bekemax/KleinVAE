from typing import Tuple, Type

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from src.models.components.vae import SimpleVAE
from src.models.vae_module import VAEModule
from src.utils.topology_utils import project_to_torus


class TorusVAEModule(VAEModule):
    def __init__(
        self,
        model: SimpleVAE,
        optimizer: Type[Optimizer],
        scheduler: LRScheduler | None = None,
        prior_std: float = 0.1,
        lr: float = 1e-3,
        kl_weight: float = 1e-1,
    ) -> None:
        super().__init__(
            model=model,
            latent_dim=model.latent_dim,
            optimizer=optimizer,
            scheduler=scheduler,
            lr=lr,
            kl_weight=kl_weight,
        )
        self.prior_std = prior_std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.model.encode(x)
        z_on_plane = self.model.reparameterize(mu, log_var)
        z_on_torus = project_to_torus(z_on_plane, stack=True)
        recon_x = self.model.decode(z_on_torus)  # type: ignore[arg-type]
        return recon_x, mu, log_var

    def kl_loss(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        prior_var = self.prior_std**2
        prior_log_var = torch.log(mu.new_tensor(prior_var))
        kl_per_dim = prior_log_var - log_var + (log_var.exp() + (mu - 0.5).pow(2)) / prior_var - 1
        return 0.5 * torch.sum(kl_per_dim, dim=1).mean()

    def vae_loss(
        self, x: torch.Tensor, recon_x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.binary_cross_entropy(recon_x, x.view(x.size(0), -1), reduction="mean")
        kl_div = self.kl_loss(mu, log_var)
        return recon_loss + self.kl_weight * kl_div, recon_loss, kl_div


if __name__ == "__main__":
    model = SimpleVAE(input_dim=28 * 28, hidden_dims=[128, 64], latent_dim=2)
    vae_module = TorusVAEModule(model=model, optimizer=torch.optim.Adam)

    x = torch.randn(16, 28 * 28)
    recon_x, mu, log_var = vae_module(x)
    print(f"Reconstructed x shape: {recon_x.shape}, mu shape: {mu.shape}, log_var shape: {log_var.shape}")
