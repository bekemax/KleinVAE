import lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from argparse import Namespace
from typing import Any, Dict, Tuple, Type

from src.models.components.vae import SimpleVAE


class VanillaVAEModule(pl.LightningModule):
    hparams: Namespace

    def __init__(
        self,
        model: SimpleVAE,
        latent_dim: int,
        optimizer: Type[Optimizer],
        scheduler: LRScheduler | None = None,
        lr: float = 1e-3,
        kl_weight: float = 1.0,
    ):
        super().__init__()
        # Using save_hyperparameters() automatically logs these values
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.latent_dim = latent_dim
        self.lr = lr
        self.kl_weight = kl_weight

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Standard reparameterization trick: z = mu + epsilon * std."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Defines the forward pass of the VAE."""
        # The encoder must output a tensor of shape [batch, 2 * latent_dim]
        mu, log_var, _ = self.model.encode(x)
        print(mu.shape, log_var.shape)

        z = self.reparameterize(mu, log_var)
        recon_x = self.model.decode(z)
        return recon_x, mu, log_var

    def _vae_loss(
        self, x: torch.Tensor, recon_x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates the VAE loss."""
        # Reconstruction Loss (using MSE here, can be changed to BCE for binary data)
        recon_loss = F.mse_loss(recon_x, x.view(x.size(0), -1), reduction="sum") / x.shape[0]

        # KL Divergence
        # analytical formula for KL(N(mu, sigma^2) || N(0, 1))
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        kl_div = kl_div.mean()

        total_loss = recon_loss + self.kl_weight * kl_div
        return total_loss, recon_loss, kl_div

    def training_step(self, batch, batch_idx):
        x, _ = batch  # Assuming batch is a tuple (data, labels)
        x = x.view(x.size(0), -1)  # Flatten input
        recon_x, mu, log_var = self.forward(x)
        loss, recon_loss, kl = self._vae_loss(x, recon_x, mu, log_var)

        self.log_dict({"train_loss": loss, "train_recon_loss": recon_loss, "train_kl_div": kl}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        recon_x, mu, log_var = self.forward(x)
        loss, recon_loss, kl = self._vae_loss(x, recon_x, mu, log_var)

        self.log_dict({"val_loss": loss, "val_recon_loss": recon_loss, "val_kl_div": kl})
        return loss

    def sample(self, num_samples: int) -> torch.Tensor:
        """Generate new samples from the latent space."""
        device = self.device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.model.decode(z)
        return samples

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())  # type: ignore
        if self.hparams.scheduler is not None and self.hparams.scheduler != {}:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": True,
                    "name": "lr",
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    latent_dim = 4
    model = SimpleVAE(input_dim=28 * 28, hidden_dims=[128, 64], latent_dim=latent_dim)
    vae_module = VanillaVAEModule(model=model, latent_dim=latent_dim, optimizer=torch.optim.Adam)

    x = torch.randn(16, 28 * 28)
    recon_x, mu, log_var = vae_module(x)
    print(f"Reconstructed x shape: {recon_x.shape}, mu shape: {mu.shape}, log_var shape: {log_var.shape}")
