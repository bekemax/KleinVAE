from src.utils.topology_utils import project_to_torus
from .components.vae import SimpleVAE

import lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.distributions.kl import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Uniform

import numpy as np
from ripser import ripser
from persim import bottleneck

from argparse import Namespace
from typing import Any, Dict, Tuple, Type, Union, Optional


class TorusVAEModule(pl.LightningModule):
    hparams: Namespace

    def __init__(
        self,
        model: SimpleVAE,
        optimizer: Type[Optimizer],
        scheduler: LRScheduler | None = None,
        sigma2: float = 5,
        lr: float = 1e-3,
        batch_size: int = 1,
        kl_weight: float = 1e-1,
        topo_metric_freq: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.kl_weight = kl_weight
        self.sigma2 = sigma2
        self.topo_metric_freq = topo_metric_freq

    def sample(self, num_samples, return_unprojected: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        prior_dist = Uniform(low=0, high=1)
        samples = prior_dist.sample([num_samples, 2])
        if return_unprojected:
            return project_to_torus(samples, stack=True), samples  # type: ignore
        else:
            return project_to_torus(samples, stack=True)

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
        z_on_torus = project_to_torus(z_on_plane, stack=True)

        recon_x = self.decode(z_on_torus)  # type: ignore

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

    def on_validation_epoch_end(self):
        """
        Calculates topological metrics, but only every N epochs.

        ```To reduce computational cost, topological similarity metrics (bottleneck distance) were computed every 10 validation epochs.```

        """
        is_last_epoch = self.current_epoch == self.trainer.max_epochs - 1  # to always compute at the end
        if (self.current_epoch + 1) % self.topo_metric_freq == 0 or is_last_epoch:
            print(f"\n--- Epoch {self.current_epoch}: Calculating topological metrics ---")

            # 1. Generate reconstructions
            with torch.no_grad():
                recon_x, _, _ = self.forward(self.trainer.datamodule.data_for_pd.to(self.device))

            # 2. Compute PDs and Bottleneck Distances
            reconstructed_pd_over_2 = ripser(recon_x, maxdim=2, coeff=2)["dgms"]
            reconstructed_pd_over_3 = ripser(recon_x, maxdim=2, coeff=3)["dgms"]

            bottlenecks_over_2 = np.array(
                [bottleneck(self.trainer.datamodule.original_pd_over_2[i], reconstructed_pd_over_2[i]) for i in range(3)]
            )
            bottlenecks_over_3 = np.array(
                [bottleneck(self.trainer.datamodule.original_pd_over_3[i], reconstructed_pd_over_3[i]) for i in range(3)]
            )

            total_dist_over_2 = np.linalg.norm(bottlenecks_over_2).__float__()
            total_dist_over_3 = np.linalg.norm(bottlenecks_over_3).__float__()

            # 3. Log the metrics
            for i in range(3):
                self.log(f"bottleneck_over_2_dim_{i}", bottlenecks_over_2[i], prog_bar=True)
                self.log(f"bottleneck_over_3_dim_{i}", bottlenecks_over_3[i], prog_bar=True)
            self.log("total_bottleneck_over_2", total_dist_over_2, prog_bar=True)
            self.log("total_bottleneck_over_3", total_dist_over_3, prog_bar=True)

            print(f"--- Finished topological metrics. Total Distances = {total_dist_over_2:.4f}, {total_dist_over_3:.4f} ---\n")

        if is_last_epoch:
            print("Caching final metric and diagram for on_train_end.")
            self.final_reconstructed_pd_over_2 = reconstructed_pd_over_2
            self.final_reconstructed_pd_over_3 = reconstructed_pd_over_3
            self.final_bottlenecks_over_2 = bottlenecks_over_2
            self.final_bottlenecks_over_3 = bottlenecks_over_3
            self.final_total_dist_over_2 = total_dist_over_2
            self.final_total_dist_over_3 = total_dist_over_3

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
    model = SimpleVAE(input_dim=28 * 28, hidden_dims=[128, 64], latent_dim=2)
    vae_module = TorusVAEModule(model=model, optimizer=torch.optim.Adam)

    x = torch.randn(16, 28 * 28)
    recon_x, mu, log_var = vae_module(x)
    print(f"Reconstructed x shape: {recon_x.shape}, mu shape: {mu.shape}, log_var shape: {log_var.shape}")
