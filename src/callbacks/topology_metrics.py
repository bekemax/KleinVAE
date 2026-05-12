from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from matplotlib import pyplot as plt

from src.utils.topology_utils import (
    compute_pairwise_bottlenecks,
    compute_persistence_diagrams,
    plot_persistence_diagram,
)


class TopologyMetricsCallback(Callback):
    def __init__(
        self,
        every_n_epochs: int = 10,
        field_orders: Sequence[int] = (2, 3),
        save_figures: bool = True,
        save_artifacts: bool = True,
    ) -> None:
        self.every_n_epochs = every_n_epochs
        self.field_orders = tuple(field_orders)
        self.save_figures = save_figures
        self.save_artifacts = save_artifacts
        self.original_pds: Optional[dict[int, list[np.ndarray]]] = None
        self.final_reconstructed_pds: Optional[dict[int, list[np.ndarray]]] = None
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

    def _save_reconstructed_pd_figures(
        self,
        trainer: Trainer,
        reconstructed_diagrams: dict[int, list[np.ndarray]],
    ) -> None:
        if not self.save_figures:
            return

        epoch = trainer.current_epoch + 1
        artifact_dir = Path(trainer.default_root_dir) / "persistence_diagrams"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, len(self.field_orders), figsize=(6 * len(self.field_orders), 6), squeeze=False)
        for ax, coeff in zip(axes[0], self.field_orders):
            plot_persistence_diagram(
                diagram=reconstructed_diagrams[coeff],
                title=rf"Reconstructed PD over $\mathbb{{Z}}_{coeff}$ at Epoch {epoch}",
                ax=ax,
            )
        plt.tight_layout()
        figure_path = artifact_dir / f"reconstructed_pd_epoch_{epoch:03d}.png"
        fig.savefig(figure_path)
        plt.close(fig)

        wandb_logger = self._wandb_logger(trainer)
        if wandb_logger is not None:
            wandb_logger.log_image("reconstructed_pd", images=[str(figure_path)], caption=[f"Epoch {epoch}"])

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

        log_data = self._generate_log_data(trainer, pl_module)
        reconstructed_diagrams = log_data["reconstructed_diagrams"]
        bottlenecks = log_data["bottlenecks"]
        total_bottlenecks = log_data["total_bottlenecks"]

        for k, distances in bottlenecks.items():
            for i, distance in enumerate(distances):
                pl_module.log(f"bottleneck_over_{k}_dim_{i}", float(distance), prog_bar=True)
            pl_module.log(f"total_bottleneck_over_{k}", total_bottlenecks[k], prog_bar=True)

        self._save_reconstructed_pd_figures(trainer, reconstructed_diagrams)

        totals_msg = ", ".join(f"Z/{coeff}: {total_bottlenecks[coeff]:.4f}" for coeff in self.field_orders)
        pl_module.print(f"--- Finished topological metrics. Total Distances = {totals_msg} ---\n")

        if is_last_epoch:
            self.final_reconstructed_pds = reconstructed_diagrams
            self.final_bottlenecks = bottlenecks
            self.final_total_dist = total_bottlenecks

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.global_rank != 0 or not self.save_artifacts or not self._has_eval_data(trainer):
            return

        if self.final_reconstructed_pds is None:
            log_data = self._generate_log_data(trainer, pl_module)
            self.final_reconstructed_pds = log_data["reconstructed_diagrams"]
            self.final_bottlenecks = log_data["bottlenecks"]
            self.final_total_dist = log_data["total_bottlenecks"]

        if self.final_bottlenecks is None or self.final_total_dist is None:
            return

        artifact_dir = Path(trainer.default_root_dir) / "persistence_diagrams"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        original_pds = self._get_original_diagrams(trainer)
        self._save_final_pd_comparison_figure(artifact_dir, original_pds, self.final_reconstructed_pds)

        for coeff in self.field_orders:
            np.savez_compressed(artifact_dir / f"original_pd_z{coeff}.npz", *original_pds[coeff])
            np.savez_compressed(artifact_dir / f"reconstructed_pd_z{coeff}.npz", *self.final_reconstructed_pds[coeff])

        wandb_logger = self._wandb_logger(trainer)
        if wandb_logger is not None:
            wandb_logger.experiment.log_artifact(
                str(artifact_dir),
                name=f"pd_results_{wandb_logger.experiment.id}",
                type="persistence_diagrams",
            )

        pl_module.print(f"Final persistence artifacts saved to {artifact_dir}.")
