from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from matplotlib import pyplot as plt

from src.models.klein_vae_module import KleinVAEModule
from src.models.torus_vae_module import TorusVAEModule
from src.utils.topology_utils import (
    compute_pairwise_bottlenecks,
    compute_persistence_diagrams,
    klein_distance_matrix,
    plot_persistence_diagram,
    project_to_klein,
    project_to_torus,
    torus_distance_matrix,
)


class TopologyMetricsCallback(Callback):
    def __init__(
        self,
        every_n_epochs: int = 10,
        field_orders: Sequence[int] = (2, 3),
        save_figures: bool = True,
        save_artifacts: bool = True,
        log_reconstruction_pd: bool = True,
        log_latent_pd: bool = True,
    ) -> None:
        self.every_n_epochs = every_n_epochs
        self.field_orders = tuple(field_orders)
        self.save_figures = save_figures
        self.save_artifacts = save_artifacts
        self.log_reconstruction_pd = log_reconstruction_pd
        self.log_latent_pd = log_latent_pd
        self.original_pds: Optional[dict[int, list[np.ndarray]]] = None
        self.final_reconstructed_pds: Optional[dict[int, list[np.ndarray]]] = None
        self.final_latent_pds: Optional[dict[int, list[np.ndarray]]] = None
        self.final_bottlenecks: Optional[dict[int, np.ndarray]] = None
        self.final_total_dist: Optional[dict[int, float]] = None

    def _has_eval_data(self, trainer: Trainer) -> bool:
        datamodule = trainer.datamodule  # type: ignore[union-attr]
        return datamodule is not None and hasattr(datamodule, "data_for_pd")

    def _eval_input(self, trainer: Trainer, pl_module: LightningModule) -> torch.Tensor:
        return trainer.datamodule.data_for_pd.to(pl_module.device)  # type: ignore[union-attr]

    def _wandb_logger(self, trainer: Trainer) -> Optional[WandbLogger]:
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                return logger
        return None

    def _generate_reconstructions(self, trainer: Trainer, pl_module: LightningModule) -> torch.Tensor:
        with torch.no_grad():
            recon_x, *_ = pl_module(self._eval_input(trainer, pl_module))
        return recon_x.detach().cpu()

    def _generate_latent_diagrams(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> Optional[dict[int, list[np.ndarray]]]:
        if not self.log_latent_pd:
            return None

        with torch.no_grad():
            x = self._eval_input(trainer, pl_module)
            mu, _ = pl_module.model.encode(x)  # type: ignore[attr-defined]

            if isinstance(pl_module, KleinVAEModule):
                latent_points = project_to_klein(mu)
                distance_matrix = klein_distance_matrix(latent_points)
                return compute_persistence_diagrams(distance_matrix, coeffs=self.field_orders, distance_matrix=True)

            if isinstance(pl_module, TorusVAEModule):
                latent_points = project_to_torus(mu, stack=True)
                distance_matrix = torus_distance_matrix(latent_points)
                return compute_persistence_diagrams(distance_matrix, coeffs=self.field_orders, distance_matrix=True)

            return compute_persistence_diagrams(mu, coeffs=self.field_orders)

    def _get_original_diagrams(self, trainer: Trainer) -> dict[int, list[np.ndarray]]:
        if self.original_pds is None:
            data_for_pd = trainer.datamodule.data_for_pd.detach().cpu()  # type: ignore[union-attr]
            self.original_pds = compute_persistence_diagrams(data_for_pd, coeffs=list(self.field_orders))
        return self.original_pds

    def _generate_log_data(self, trainer: Trainer, pl_module: LightningModule) -> dict:
        original_diagrams = self._get_original_diagrams(trainer)
        recon_x = self._generate_reconstructions(trainer, pl_module)
        reconstructed_diagrams = compute_persistence_diagrams(recon_x, coeffs=list(self.field_orders))
        bottlenecks = compute_pairwise_bottlenecks(original_diagrams, reconstructed_diagrams)
        total_bottlenecks = {k: float(np.linalg.norm(v)) for k, v in bottlenecks.items()}
        return {
            "reconstructed_diagrams": reconstructed_diagrams,
            "bottlenecks": bottlenecks,
            "total_bottlenecks": total_bottlenecks,
        }

    def _save_pd_figures(
        self,
        trainer: Trainer,
        diagrams: dict[int, list[np.ndarray]],
        *,
        wandb_key: str,
        title_prefix: str,
        filename_prefix: str,
    ) -> None:
        if not self.save_figures:
            return

        epoch = trainer.current_epoch + 1
        artifact_dir = Path(trainer.default_root_dir) / "persistence_diagrams"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, len(self.field_orders), figsize=(6 * len(self.field_orders), 6), squeeze=False)
        for ax, coeff in zip(axes[0], self.field_orders):
            plot_persistence_diagram(
                diagram=diagrams[coeff],
                title=rf"{title_prefix} over $\mathbb{{Z}}_{coeff}$ at Epoch {epoch}",
                ax=ax,
            )
        plt.tight_layout()
        figure_path = artifact_dir / f"{filename_prefix}_epoch_{epoch:03d}.png"
        fig.savefig(figure_path)
        plt.close(fig)

        wandb_logger = self._wandb_logger(trainer)
        if wandb_logger is not None:
            wandb_logger.log_image(wandb_key, images=[str(figure_path)], caption=[f"Epoch {epoch}"])

    def _save_final_pd_comparison_figure(
        self,
        artifact_dir: Path,
        original_pds: dict[int, list[np.ndarray]],
        reconstructed_pds: dict[int, list[np.ndarray]],
    ) -> Path:
        fig, axes = plt.subplots(2, len(self.field_orders), figsize=(6 * len(self.field_orders), 12), squeeze=False)
        fig.suptitle("Final Persistence Diagram Comparison", fontsize=16)

        for axis_idx, coeff in enumerate(self.field_orders):
            plot_persistence_diagram(original_pds[coeff], title=rf"Original PD over $\mathbb{{Z}}_{coeff}$", ax=axes[0, axis_idx])
            plot_persistence_diagram(
                reconstructed_pds[coeff],
                title=rf"Reconstructed PD over $\mathbb{{Z}}_{coeff}$",
                ax=axes[1, axis_idx],
            )

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plot_path = artifact_dir / "final_pd_comparison.png"
        fig.savefig(plot_path)
        plt.close(fig)
        return plot_path

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking or not self._has_eval_data(trainer):
            return

        is_last_epoch = trainer.current_epoch == trainer.max_epochs - 1
        should_compute = ((trainer.current_epoch + 1) % self.every_n_epochs == 0) or is_last_epoch
        if not should_compute:
            return

        pl_module.print(f"\n--- Epoch {trainer.current_epoch}: Calculating topological metrics ---")

        if self.log_reconstruction_pd:
            log_data = self._generate_log_data(trainer, pl_module)
            reconstructed_diagrams = log_data["reconstructed_diagrams"]
            bottlenecks = log_data["bottlenecks"]
            total_bottlenecks = log_data["total_bottlenecks"]

            for k, distances in bottlenecks.items():
                for i, distance in enumerate(distances):
                    pl_module.log(f"bottleneck_over_{k}_dim_{i}", float(distance), prog_bar=True)
                pl_module.log(f"total_bottleneck_over_{k}", total_bottlenecks[k], prog_bar=True)

            self._save_pd_figures(
                trainer,
                reconstructed_diagrams,
                wandb_key="reconstructed_pd",
                title_prefix="Reconstructed PD",
                filename_prefix="reconstructed_pd",
            )

            totals_msg = ", ".join(f"Z/{coeff}: {total_bottlenecks[coeff]:.4f}" for coeff in self.field_orders)
            pl_module.print(f"--- Finished reconstruction topological metrics. Total Distances = {totals_msg} ---")

            if is_last_epoch:
                self.final_reconstructed_pds = reconstructed_diagrams
                self.final_bottlenecks = bottlenecks
                self.final_total_dist = total_bottlenecks

        latent_diagrams = self._generate_latent_diagrams(trainer, pl_module)
        if latent_diagrams is not None:
            self._save_pd_figures(
                trainer,
                latent_diagrams,
                wandb_key="latent_pd",
                title_prefix="Latent PD",
                filename_prefix="latent_pd",
            )
            pl_module.print("--- Logged latent persistence diagrams ---\n")
            if is_last_epoch:
                self.final_latent_pds = latent_diagrams

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.global_rank != 0 or not self.save_artifacts or not self._has_eval_data(trainer):
            return

        if self.log_reconstruction_pd and self.final_reconstructed_pds is None:
            log_data = self._generate_log_data(trainer, pl_module)
            self.final_reconstructed_pds = log_data["reconstructed_diagrams"]
            self.final_bottlenecks = log_data["bottlenecks"]
            self.final_total_dist = log_data["total_bottlenecks"]

        if self.log_latent_pd and self.final_latent_pds is None:
            self.final_latent_pds = self._generate_latent_diagrams(trainer, pl_module)

        if self.final_reconstructed_pds is None and self.final_latent_pds is None:
            return

        artifact_dir = Path(trainer.default_root_dir) / "persistence_diagrams"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        original_pds = self._get_original_diagrams(trainer)
        if self.final_reconstructed_pds is not None:
            self._save_final_pd_comparison_figure(artifact_dir, original_pds, self.final_reconstructed_pds)

        for coeff in self.field_orders:
            np.savez_compressed(artifact_dir / f"original_pd_z{coeff}.npz", *original_pds[coeff])
            if self.final_reconstructed_pds is not None:
                np.savez_compressed(artifact_dir / f"reconstructed_pd_z{coeff}.npz", *self.final_reconstructed_pds[coeff])
            if self.final_latent_pds is not None:
                np.savez_compressed(artifact_dir / f"latent_pd_z{coeff}.npz", *self.final_latent_pds[coeff])

        wandb_logger = self._wandb_logger(trainer)
        if wandb_logger is not None:
            wandb_logger.experiment.log_artifact(
                str(artifact_dir),
                name=f"pd_results_{wandb_logger.experiment.id}",
                type="persistence_diagrams",
            )

        pl_module.print(f"Final persistence artifacts saved to {artifact_dir}.")
