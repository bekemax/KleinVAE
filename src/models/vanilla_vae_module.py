import lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import wandb

import numpy as np
from matplotlib import pyplot as plt

from argparse import Namespace
from typing import Any, Dict, Tuple, Type

from src.models.components.vae import SimpleVAE
from src.utils.topology_utils import compute_pairwise_bottlenecks, compute_persistence_diagrams, plot_persistence_diagram


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
        topo_metric_freq: int = 10,
    ):
        super().__init__()
        # Using save_hyperparameters() automatically logs these values
        self.hparams.hidden_dims = model.hidden_dims
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.latent_dim = latent_dim
        self.lr = lr
        self.kl_weight = kl_weight
        self.topo_metric_freq = topo_metric_freq

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Standard reparameterization trick: z = mu + epsilon * std."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Defines the forward pass of the VAE."""
        # The encoder must output a tensor of shape [batch, 2 * latent_dim]
        mu, log_var, _ = self.model.encode(x)

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
        # analytical formula for KL(N(mu, sigma^2) || N(0, p) where p = N(0, prior_scale^2 * I)
        prior_var = 0.1**2  # prior_scale**2
        kl_div = -0.5 * torch.sum(1 + torch.log(torch.tensor(prior_var)) + log_var - (mu.pow(2) + log_var.exp()) / prior_var, dim=1)
        kl_div = kl_div.mean()

        total_loss = recon_loss + self.kl_weight * kl_div
        return total_loss, recon_loss, kl_div

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x = x.view(x.size(0), -1)  # Flatten input
        recon_x, mu, log_var = self.forward(x)
        loss, recon_loss, kl = self._vae_loss(x, recon_x, mu, log_var)

        self.log_dict({"train_loss": loss, "recon_loss": recon_loss, "kl_div": kl}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
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

    def _generate_reconstructions(self):
        with torch.no_grad():
            recon_x, _, _ = self.forward(self.trainer.datamodule.data_for_pd.to(self.device))
        return recon_x

    def _generate_log_data(self):
        recon_x = self._generate_reconstructions()
        reconstructed_diagrams = compute_persistence_diagrams(recon_x)
        bottlenecks = compute_pairwise_bottlenecks(self.trainer.datamodule.original_pds, reconstructed_diagrams)
        total_bottlenecks = {k: np.linalg.norm(v).__float__() for k, v in bottlenecks.items()}
        return {"reconstructed_diagrams": reconstructed_diagrams, "bottlenecks": bottlenecks, "total_bottlenecks": total_bottlenecks}

    def on_validation_epoch_end(self):
        """
        Calculates topological metrics, but only every N epochs.

        ```To reduce computational cost, topological similarity metrics (bottleneck distance) were computed every 10 validation epochs.```

        """
        is_last_epoch = self.current_epoch == self.trainer.max_epochs - 1  # to always compute at the end
        if (self.current_epoch + 1) % self.topo_metric_freq == 0 or is_last_epoch:
            print(f"\n--- Epoch {self.current_epoch}: Calculating topological metrics ---")

            log_data = self._generate_log_data()
            reconstructed_diagrams = log_data["reconstructed_diagrams"]
            bottlenecks = log_data["bottlenecks"]
            total_bottlenecks = log_data["total_bottlenecks"]

            # 3. Log the metrics
            for k in bottlenecks.keys():
                for i in range(3):
                    self.log(f"bottleneck_over_{k}_dim_{i}", bottlenecks[k][i], prog_bar=True)
                self.log(f"total_bottleneck_over_{k}", total_bottlenecks[k], prog_bar=True)

            # 4. Log the PDs
            title2 = "Reconstructed PD over $\mathbb{Z}_2$ at Epoch " + str(self.current_epoch + 1)
            title3 = "Reconstructed PD over $\mathbb{Z}_3$ at Epoch " + str(self.current_epoch + 1)
            fig2, _ = plot_persistence_diagram(diagram=reconstructed_diagrams[2], title=title2)  # type: ignore
            fig3, _ = plot_persistence_diagram(diagram=reconstructed_diagrams[3], title=title3)  # type: ignore
            self.logger.experiment.log(
                {"reconstructed_pd_z2": wandb.Image(fig2, caption=title2), "reconstructed_pd_z3": wandb.Image(fig3, caption=title3)}
            )
            plt.close(fig2)
            plt.close(fig3)

            print(f"--- Finished topological metrics. Total Distances = {total_bottlenecks[2]:.4f}, {total_bottlenecks[3]:.4f} ---\n")

            # 4.5 Computing var of latent codes
            with torch.no_grad():
                mu, log_var, _ = self.model.encode(self.trainer.datamodule.data_for_pd.to(self.device))
                z = self.reparameterize(mu, log_var)
                var_z = torch.sum(torch.var(z, dim=0))
                self.log("var_z", var_z, prog_bar=True)

        if is_last_epoch:
            print("Caching final metric and diagram for on_train_end.")
            self.final_reconstructed_pds = reconstructed_diagrams
            self.final_bottlenecks = bottlenecks
            self.final_total_dist = total_bottlenecks

    def on_train_end(self) -> None:
        # Ensure this only runs on the main process to avoid duplication
        if self.trainer.global_rank != 0:
            return

        print("\n--- Training finished. Generating final metrics and artifacts ---")
        # 1. Reuse cached data (with a fallback to recompute if needed)
        # This assumes you cached these in on_validation_epoch_end
        if not hasattr(self, "final_reconstructed_pds") or self.final_reconstructed_pds is None:
            print("Final diagrams not cached. Re-computing now as a fallback...")
            log_data = self._generate_log_data()
            self.final_reconstructed_pds = log_data["reconstructed_diagrams"]
            self.final_bottlenecks = log_data["bottlenecks"]
            self.final_total_dist = log_data["total_bottlenecks"]

        # 2. Log final bottleneck distances
        for k, distances in self.final_bottlenecks.items():
            for i, dist in enumerate(distances):
                self.logger.experiment.summary[f"final_bottleneck_over_{k}_dim_{i}"] = dist
            self.logger.experiment.summary[f"final_total_bottleneck_over_{k}"] = self.final_total_dist[k]

        # 3. Log final PD plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle("Final Persistence Diagram Comparison", fontsize=16)

        plot_persistence_diagram(self.trainer.datamodule.original_pds[2], title="Original PD over $\mathbb{Z}_2$", ax=axes[0, 0])
        plot_persistence_diagram(self.final_reconstructed_pds[2], title="Reconstructed PD over $\mathbb{Z}_2$", ax=axes[0, 1])
        plot_persistence_diagram(self.trainer.datamodule.original_pds[3], title="Original PD over $\mathbb{Z}_3$", ax=axes[1, 0])
        plot_persistence_diagram(self.final_reconstructed_pds[3], title="Reconstructed PD over $\mathbb{Z}_3$", ax=axes[1, 1])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = "final_pd_comparison.png"
        fig.savefig("data/" + plot_filename)
        plt.close(fig)

        np.savez_compressed("data/original_pd_z2.npz", *self.trainer.datamodule.original_pds[2])
        np.savez_compressed("data/original_pd_z3.npz", *self.trainer.datamodule.original_pds[3])
        np.savez_compressed("data/reconstructed_pd_z2.npz", *self.final_reconstructed_pds[2])
        np.savez_compressed("data/reconstructed_pd_z3.npz", *self.final_reconstructed_pds[3])

        artifact = wandb.Artifact(name=f"pd_results_{self.logger.experiment.id}", type="persistence_diagrams")
        artifact.add_file("data/original_pd_z2.npz")
        artifact.add_file("data/original_pd_z3.npz")
        artifact.add_file("data/reconstructed_pd_z2.npz")
        artifact.add_file("data/reconstructed_pd_z3.npz")
        artifact.add_file("data/" + plot_filename)

        self.logger.experiment.log_artifact(artifact)
        print("--- Final metrics and artifacts have been saved to W&B. ---")

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
