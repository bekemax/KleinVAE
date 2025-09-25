from src.utils.topology_utils import (
    compute_pairwise_bottlenecks,
    compute_persistence_diagrams,
    plot_persistence_diagram,
    project_to_klein,
    project_to_torus,
)
from .components.vae import SimpleVAE

import lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.distributions.kl import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Uniform

import wandb

import numpy as np
from matplotlib import pyplot as plt

from argparse import Namespace
from typing import Any, Dict, List, Tuple, Type, Union, Optional


class KleinVAEModule(pl.LightningModule):
    hparams: Namespace

    def __init__(
        self,
        model: SimpleVAE,
        optimizer: Type[Optimizer],
        sigma2: float = 5,
        lr: float = 1e-3,
        batch_size: int = 1,
        kl_weight: float = 1e-1,
        scheduler: LRScheduler | None = None,
        topo_metric_freq: int = 10,
    ):
        super().__init__()
        self.hparams.hidden_dims = model.hidden_dims
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.kl_weight = kl_weight
        self.sigma2 = sigma2
        self.topo_metric_freq = topo_metric_freq

        self.final_reconstructed_pds: Optional[Dict[int, List[np.ndarray]]] = None
        self.final_bottlenecks: Optional[Dict[int, List[float]]] = None
        self.final_total_dist: Optional[Dict[int, float]] = None

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

    def _generate_reconstructions(self):
        with torch.no_grad():
            recon_x, mu, L = self.forward(self.trainer.datamodule.data_for_pd.to(self.device))
        return recon_x, mu, L

    def _generate_log_data(self):
        recon_x, _, _ = self._generate_reconstructions()
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
        if ((self.current_epoch + 1) % self.topo_metric_freq == 0) or is_last_epoch:
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
                mu, log_sigma, non_diag = self.encode(self.trainer.datamodule.data_for_pd.to(self.device))
                z_on_plane, _ = self.reparameterize(mu, log_sigma, non_diag)
                z_on_torus = project_to_torus(z_on_plane, stack=True)
                var_z = torch.sum(torch.var(z_on_torus, dim=0))
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
    model = SimpleVAE(input_dim=28 * 28, hidden_dims=[128, 64])
    vae_module = KleinVAEModule(model=model, optimizer=torch.optim.Adam)

    x = torch.randn(16, 28 * 28)
    recon_x, mu, log_var = vae_module(x)
    print(f"Reconstructed x shape: {recon_x.shape}, mu shape: {mu.shape}, log_var shape: {log_var.shape}")
