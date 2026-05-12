from pathlib import Path
from typing import Optional

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
        save_figures: bool = True,
        save_artifacts: bool = True,
    ) -> None:
        self.every_n_epochs = every_n_epochs
        self.save_figures = save_figures
        self.save_artifacts = save_artifacts
        self.final_reconstructed_pds: Optional[dict[int, list[np.ndarray]]] = None
        self.final_bottlenecks: Optional[dict[int, np.ndarray]] = None
        self.final_total_dist: Optional[dict[int, float]] = None

    def _has_eval_data(self, trainer: Trainer) -> bool:
        datamodule = trainer.datamodule  # type: ignore[union-attr]
        return datamodule is not None and hasattr(datamodule, "data_for_pd") and hasattr(datamodule, "original_pds")

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

    def _generate_log_data(self, trainer: Trainer, pl_module: LightningModule) -> dict:
        recon_x = self._generate_reconstructions(trainer, pl_module)
        reconstructed_diagrams = compute_persistence_diagrams(recon_x)
        bottlenecks = compute_pairwise_bottlenecks(
            trainer.datamodule.original_pds,  # type: ignore[union-attr]
            reconstructed_diagrams,  # type: ignore[union-attr]
        )
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

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        plot_persistence_diagram(  # type: ignore
            diagram=reconstructed_diagrams[2],
            title=rf"Reconstructed PD over $\mathbb{{Z}}_2$ at Epoch {epoch}",
            ax=axes[0],
        )
        plot_persistence_diagram(  # type: ignore
            diagram=reconstructed_diagrams[3],
            title=rf"Reconstructed PD over $\mathbb{{Z}}_3$ at Epoch {epoch}",
            ax=axes[1],
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
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle("Final Persistence Diagram Comparison", fontsize=16)

        plot_persistence_diagram(original_pds[2], title=r"Original PD over $\mathbb{Z}_2$", ax=axes[0, 0])
        plot_persistence_diagram(reconstructed_pds[2], title=r"Reconstructed PD over $\mathbb{Z}_2$", ax=axes[0, 1])
        plot_persistence_diagram(original_pds[3], title=r"Original PD over $\mathbb{Z}_3$", ax=axes[1, 0])
        plot_persistence_diagram(reconstructed_pds[3], title=r"Reconstructed PD over $\mathbb{Z}_3$", ax=axes[1, 1])

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

        pl_module.print(f"--- Finished topological metrics. Total Distances = {total_bottlenecks[2]:.4f}, {total_bottlenecks[3]:.4f} ---\n")

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

        original_pds = trainer.datamodule.original_pds  # type: ignore[union-attr]
        plot_path = self._save_final_pd_comparison_figure(artifact_dir, original_pds, self.final_reconstructed_pds)

        artifact_paths = [
            artifact_dir / "original_pd_z2.npz",
            artifact_dir / "original_pd_z3.npz",
            artifact_dir / "reconstructed_pd_z2.npz",
            artifact_dir / "reconstructed_pd_z3.npz",
            plot_path,
        ]
        np.savez_compressed(artifact_paths[0], *original_pds[2])
        np.savez_compressed(artifact_paths[1], *original_pds[3])
        np.savez_compressed(artifact_paths[2], *self.final_reconstructed_pds[2])
        np.savez_compressed(artifact_paths[3], *self.final_reconstructed_pds[3])

        wandb_logger = self._wandb_logger(trainer)
        if wandb_logger is not None:
            wandb_logger.experiment.log_artifact(
                str(artifact_dir),
                name=f"pd_results_{wandb_logger.experiment.id}",
                type="persistence_diagrams",
            )

        pl_module.print(f"Final persistence artifacts saved to {artifact_dir}.")
