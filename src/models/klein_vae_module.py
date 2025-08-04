from src.utils import project_to_klein
from .components.vae import SimpleVAE

import lightning as pl
import torch
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Uniform
from torch.optim.lr_scheduler import ReduceLROnPlateau

from typing import Tuple, Union, Optional


class KleinVAEModule(pl.LightningModule):
    def __init__(self, model: SimpleVAE, sigma2: float = 5, lr: float = 1e-3, batch_size: int = 1, kl_weight: float = 1e-1):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.kl_weight = kl_weight
        self.sigma2 = sigma2

    def sample(self, num_samples, return_unprojected: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        prior_dist = Uniform(low=0, high=1)
        samples = prior_dist.sample([num_samples, 2])
        if return_unprojected:
            return project_to_klein(samples), samples
        else:
            return project_to_klein(samples)

    def reparameterize(self, mu, log_sigma, non_diag) -> Tuple[torch.Tensor, torch.Tensor]:
        var = torch.exp(log_sigma)
        batch_size = mu.shape[0]

        L = torch.zeros(batch_size, 2, 2, device=mu.device)
        L[:, 0, 0] = var[:, 0]
        L[:, 1, 1] = var[:, 0]
        L[:, 1, 0] = 0  # non_diag[:, 0]

        standard_normal = MultivariateNormal(torch.zeros(2), torch.eye(2))
        eps = standard_normal.sample([batch_size])  # -> shape (batch_size, 2)
        # torch.randn_like(mu) * torch.sqrt(torch.tensor(self.sigma2, device=mu.device))
        return mu + torch.matmul(L, eps.unsqueeze(-1)).squeeze(-1), L

    def refreeze_encoder(self, flag: bool = False):
        for param in self.model.encoder.parameters():
            param.requires_grad = flag

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.model.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        recon_x = self.model.decode(z)
        return recon_x

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_sigma, non_diag = self.encode(x)
        z_on_plane, L = self.reparameterize(mu, log_sigma, non_diag)
        z_on_klein = project_to_klein(z_on_plane)

        recon_x = self.decode(z_on_klein)

        return recon_x, mu, L

    def _vae_loss(self, x, recon_x, mu, L) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="mean")

        q = MultivariateNormal(mu, scale_tril=L)

        prior_loc = torch.zeros_like(mu) + 0.5  # Center the prior at (0.5, 0.5)
        prior_scale = torch.eye(2, device=mu.device).unsqueeze(0).repeat(mu.size(0), 1, 1) * self.sigma2
        prior_dist = MultivariateNormal(prior_loc, scale_tril=prior_scale)

        kl_div = kl_divergence(q, prior_dist).mean()
        return recon_loss + self.kl_weight * kl_div, recon_loss, kl_div

    def training_step(self, batch, batch_idx):
        x = batch[0]
        recon_x, mu, L = self.forward(x)
        loss, recon_loss, kl = self._vae_loss(x, recon_x, mu, L)
        self.log_dict({"train_loss": loss, "recon_loss": recon_loss, "kl_div": kl}, prog_bar=True)

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        recon_x, mu, L = self(x)
        loss, recon_loss, kl = self._vae_loss(x, recon_x, mu, L)
        self.log_dict({"val_loss": loss, "val_recon_loss": recon_loss, "val_kl_div": kl})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.99, patience=0)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
                "name": "lr",
            },
        }
