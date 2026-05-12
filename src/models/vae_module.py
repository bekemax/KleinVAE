from collections.abc import Callable
from typing import Any, Dict, Tuple

import lightning as pl
import torch
from torch.optim.lr_scheduler import LRScheduler

from src.models.components.vae import SimpleVAE


class VAEModule(pl.LightningModule):
    """Standard VAE LightningModule.

    Torus and Klein variants subclass this module and override the latent-space pieces
    while keeping the same Lightning training, validation, optimizer, and evaluation hooks.
    """

    def __init__(
        self,
        model: SimpleVAE,
        optimizer: Callable[..., torch.optim.Optimizer],
        recon_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scheduler: Callable[..., LRScheduler] | None = None,
        lr: float = 1e-3,
        kl_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "recon_loss"])
        self.model = model
        self.recon_loss = recon_loss
        self.lr = lr
        self.kl_weight = kl_weight

    def refreeze_encoder(self, flag: bool = False) -> None:
        for param in self.model.encoder.parameters():
            param.requires_grad = flag

    def reconstruction_loss(self, x: torch.Tensor, recon_x: torch.Tensor) -> torch.Tensor:
        target_x = x if x.shape == recon_x.shape else x.reshape(x.size(0), -1)
        recon_loss = self.recon_loss(recon_x, target_x)
        if getattr(self.recon_loss, "reduction", None) == "sum":
            recon_loss = recon_loss / x.shape[0]
        return recon_loss

    def kl_loss(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(mu.pow(2) + log_var.exp() - log_var - 1, dim=1).mean()

    def _vae_loss(
        self, x: torch.Tensor, recon_x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the VAE loss.

        The VAE maximizes the evidence lower bound (ELBO):

            ELBO = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z)).

        During training we minimize the negative ELBO, written as

            loss = reconstruction_loss + KL(q(z|x) || p(z)).

        The reconstruction term is the negative log-likelihood of the input under the decoder likelihood.

        Args:
            x: Original input batch.
            recon_x: Decoder reconstruction.
            mu: Encoder mean of q(z|x).
            log_var: Encoder log-variance of q(z|x).

        Returns:
            A tuple containing total loss, reconstruction loss, and KL loss.
        """

        recon_loss = self.reconstruction_loss(x, recon_x)
        kl_div = self.kl_loss(mu, log_var)
        total_loss = recon_loss + self.kl_weight * kl_div
        return total_loss, recon_loss, kl_div

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        x = batch[0]
        recon_x, mu, log_var = self(x)
        loss, recon_loss, kl = self._vae_loss(x, recon_x, mu, log_var)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        self.log("train_recon_loss", recon_loss, on_epoch=True, batch_size=x.shape[0])
        self.log("train_kl_loss", kl, on_epoch=True, batch_size=x.shape[0])

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        x = batch[0]
        recon_x, mu, log_var = self(x)
        loss, recon_loss, kl = self._vae_loss(x, recon_x, mu, log_var)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        self.log("val_recon_loss", recon_loss, on_epoch=True, batch_size=x.shape[0])
        self.log("val_kl_loss", kl, on_epoch=True, batch_size=x.shape[0])

        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())  # type: ignore[attr-defined]
        scheduler_cfg = self.hparams.get("scheduler")
        if scheduler_cfg is not None and scheduler_cfg != {}:
            scheduler: LRScheduler = scheduler_cfg(optimizer=optimizer)
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
    vae_module = VAEModule(model=model, optimizer=torch.optim.Adam, recon_loss=torch.nn.MSELoss(reduction="sum"))

    x = torch.randn(16, 28 * 28)
    recon_x, mu, log_var = vae_module(x)
    print(f"Reconstructed x shape: {recon_x.shape}, mu shape: {mu.shape}, log_var shape: {log_var.shape}")
