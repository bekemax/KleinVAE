from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from matplotlib import pyplot as plt

from src.models.klein_vae_module import KleinVAEModule
from src.models.torus_vae_module import TorusVAEModule
from src.utils.topology_utils import (
    compute_normalized_persistence_diagrams,
    compute_pairwise_bottlenecks,
    compute_persistence_diagrams,
    compute_topology_bottlenecks,
    euclidean_distance_matrix,
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
        scale_quantile: float = 0.9,
        topology_dimensions: Sequence[int] = (1, 2),
        input_metric: str = "ambient",
        compute_during_training: bool = True,
    ) -> None:
        if input_metric not in {"ambient", "intrinsic_klein"}:
            raise ValueError("input_metric must be 'ambient' or 'intrinsic_klein'")
        self.every_n_epochs = every_n_epochs
        self.field_orders = tuple(field_orders)
        self.save_figures = save_figures
        self.save_artifacts = save_artifacts
        self.log_reconstruction_pd = log_reconstruction_pd
        self.log_latent_pd = log_latent_pd
        self.scale_quantile = scale_quantile
        self.topology_dimensions = tuple(topology_dimensions)
        self.input_metric = input_metric
        # Sweeps score the best checkpoint separately, so in-training diagrams are pure overhead.
        self.compute_during_training = compute_during_training
        self.original_pds: dict[int, list[np.ndarray]] | None = None
        self.normalized_original_pds: dict[int, list[np.ndarray]] | None = None
        self.test_original_pds: dict[int, list[np.ndarray]] | None = None
        self.test_normalized_original_pds: dict[int, list[np.ndarray]] | None = None
        self.final_reconstructed_pds: dict[int, list[np.ndarray]] | None = None
        self.final_latent_pds: dict[int, list[np.ndarray]] | None = None
        self.final_bottlenecks: dict[int, np.ndarray] | None = None
        self.final_total_dist: dict[int, float] | None = None
        self.final_latent_bottlenecks: dict[int, dict[int, float]] | None = None
        self.final_latent_topology_score: float | None = None
        self.test_reconstructed_pds: dict[int, list[np.ndarray]] | None = None
        self.test_latent_pds: dict[int, list[np.ndarray]] | None = None
        self.test_bottlenecks: dict[int, np.ndarray] | None = None
        self.test_total_dist: dict[int, float] | None = None
        self.test_latent_bottlenecks: dict[int, dict[int, float]] | None = None
        self.test_latent_topology_score: float | None = None

    # Datamodule attributes holding the point cloud and its intrinsic parameters, per stage.
    STAGE_ATTRIBUTES = {
        "val": ("data_for_pd", "validation_parameters_for_pd"),
        "test": ("test_data_for_pd", "test_parameters_for_pd"),
    }

    def _stage_attributes(self, stage: str) -> tuple[str, str]:
        if stage not in self.STAGE_ATTRIBUTES:
            raise ValueError(f"Unknown stage {stage!r}; expected one of {sorted(self.STAGE_ATTRIBUTES)}")
        return self.STAGE_ATTRIBUTES[stage]

    def _required_tensor(self, datamodule: Any, attribute: str, stage: str) -> torch.Tensor:
        value = getattr(datamodule, attribute, None)
        if value is None:
            raise ValueError(f"The {stage} stage requires datamodule.{attribute}, which is missing or None")
        return value

    def _has_eval_data(self, trainer: Trainer, stage: str = "val") -> bool:
        datamodule = trainer.datamodule  # type: ignore[union-attr]
        if datamodule is None:
            return False
        data_attribute, _ = self._stage_attributes(stage)
        return getattr(datamodule, data_attribute, None) is not None

    def _eval_input(self, trainer: Trainer, pl_module: LightningModule, stage: str = "val") -> torch.Tensor:
        datamodule = trainer.datamodule  # type: ignore[union-attr]
        data_attribute, _ = self._stage_attributes(stage)
        return self._required_tensor(datamodule, data_attribute, stage).to(pl_module.device)

    def _wandb_logger(self, trainer: Trainer) -> WandbLogger | None:
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                return logger
        return None

    def _generate_reconstructions(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str = "val",
    ) -> torch.Tensor:
        with torch.no_grad():
            eval_input = self._eval_input(trainer, pl_module, stage=stage)
            if hasattr(pl_module, "reconstruct"):
                recon_x = pl_module.reconstruct(  # type: ignore[attr-defined]
                    eval_input,
                    sample=False,
                )
            else:
                # `forward` returns logits; PDs are computed on pixel intensities, so map them back to [0, 1].
                logits, *_ = pl_module(eval_input)
                recon_x = torch.sigmoid(logits)
        return recon_x.detach().cpu()

    def _empirical_latent_variances(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str = "val",
    ) -> torch.Tensor:
        """Return empirical variance of one projected latent sample per input."""

        with torch.no_grad():
            x = self._eval_input(trainer, pl_module, stage=stage)
            mu, covariance_parameters = pl_module.model.encode(x)  # type: ignore[attr-defined]
            z = pl_module.model.reparameterize(mu, covariance_parameters)  # type: ignore[attr-defined]
            projected = (
                pl_module.project_latent(z)  # type: ignore[attr-defined]
                if hasattr(pl_module, "project_latent")
                else z
            )
            variance = projected.var(dim=0, unbiased=True).sum()
        return variance

    def _generate_latent_diagrams(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str = "val",
    ) -> dict[int, list[np.ndarray]] | None:
        if not self.log_latent_pd:
            return None

        with torch.no_grad():
            x = self._eval_input(trainer, pl_module, stage=stage)
            mu, _ = pl_module.model.encode(x)  # type: ignore[attr-defined]

            if isinstance(pl_module, KleinVAEModule):
                latent_points = project_to_klein(mu)
                distance_matrix = klein_distance_matrix(latent_points)
            elif isinstance(pl_module, TorusVAEModule):
                latent_points = project_to_torus(mu, stack=True)
                distance_matrix = torus_distance_matrix(latent_points)
            else:
                distance_matrix = euclidean_distance_matrix(mu)

            return compute_normalized_persistence_diagrams(
                distance_matrix,
                coeffs=self.field_orders,
                scale_quantile=self.scale_quantile,
            )

    def _get_original_diagrams(self, trainer: Trainer, stage: str = "val") -> dict[int, list[np.ndarray]]:
        cached = self.test_original_pds if stage == "test" else self.original_pds
        if cached is not None:
            return cached

        data_attribute, _ = self._stage_attributes(stage)
        data = self._required_tensor(trainer.datamodule, data_attribute, stage).detach().cpu()  # type: ignore[union-attr]
        diagrams = compute_persistence_diagrams(data, coeffs=list(self.field_orders))
        if stage == "test":
            self.test_original_pds = diagrams
        else:
            self.original_pds = diagrams
        return diagrams

    def _get_normalized_original_diagrams(self, trainer: Trainer, stage: str = "val") -> dict[int, list[np.ndarray]]:
        cached = self.test_normalized_original_pds if stage == "test" else self.normalized_original_pds
        if cached is not None:
            return cached

        datamodule = trainer.datamodule
        data_attribute, parameter_attribute = self._stage_attributes(stage)
        if self.input_metric == "intrinsic_klein":
            intrinsic_points = self._required_tensor(datamodule, parameter_attribute, stage)
            distance_matrix = klein_distance_matrix(intrinsic_points.detach().cpu())
        else:
            data = self._required_tensor(datamodule, data_attribute, stage).detach().cpu()
            distance_matrix = euclidean_distance_matrix(data)
        diagrams = compute_normalized_persistence_diagrams(
            distance_matrix,
            coeffs=self.field_orders,
            scale_quantile=self.scale_quantile,
        )
        if stage == "test":
            self.test_normalized_original_pds = diagrams
        else:
            self.normalized_original_pds = diagrams
        return diagrams

    def _generate_log_data(self, trainer: Trainer, pl_module: LightningModule, stage: str = "val") -> dict:
        original_diagrams = self._get_original_diagrams(trainer, stage=stage)
        recon_x = self._generate_reconstructions(trainer, pl_module, stage=stage)
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
        epoch_label: str | None = None,
    ) -> None:
        if not self.save_figures:
            return

        epoch_val = trainer.current_epoch + 1
        label = epoch_label if epoch_label is not None else f"Epoch {epoch_val}"
        artifact_dir = Path(trainer.default_root_dir) / "persistence_diagrams"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, len(self.field_orders), figsize=(6 * len(self.field_orders), 6), squeeze=False)
        for ax, coeff in zip(axes[0], self.field_orders):
            plot_persistence_diagram(
                diagram=diagrams[coeff],
                title=rf"{title_prefix} over $\mathbb{{Z}}_{coeff}$ ({label})",
                ax=ax,
            )
        plt.tight_layout()
        suffix = filename_prefix if epoch_label is not None else f"{filename_prefix}_epoch_{epoch_val:03d}"
        figure_path = artifact_dir / f"{suffix}.png"
        fig.savefig(figure_path)
        plt.close(fig)

        wandb_logger = self._wandb_logger(trainer)
        if wandb_logger is not None:
            wandb_logger.log_image(wandb_key, images=[str(figure_path)], caption=[label])

    def _save_final_pd_comparison_figure(
        self,
        artifact_dir: Path,
        original_pds: dict[int, list[np.ndarray]],
        reconstructed_pds: dict[int, list[np.ndarray]],
        filename: str = "final_pd_comparison.png",
        title: str = "Final Persistence Diagram Comparison",
    ) -> Path:
        fig, axes = plt.subplots(2, len(self.field_orders), figsize=(6 * len(self.field_orders), 12), squeeze=False)
        fig.suptitle(title, fontsize=16)

        for axis_idx, coeff in enumerate(self.field_orders):
            plot_persistence_diagram(original_pds[coeff], title=rf"Original PD over $\mathbb{{Z}}_{coeff}$", ax=axes[0, axis_idx])
            plot_persistence_diagram(
                reconstructed_pds[coeff],
                title=rf"Reconstructed PD over $\mathbb{{Z}}_{coeff}$",
                ax=axes[1, axis_idx],
            )

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plot_path = artifact_dir / filename
        fig.savefig(plot_path)
        plt.close(fig)
        return plot_path

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking or not self._has_eval_data(trainer):
            return

        empirical_variance = self._empirical_latent_variances(trainer, pl_module)
        tiny = torch.finfo(empirical_variance.dtype).tiny
        pl_module.log("latent_code_variance", empirical_variance, prog_bar=False)
        pl_module.log(
            "log_latent_code_variance",
            empirical_variance.clamp_min(tiny).log(),
            prog_bar=False,
        )

        is_last_epoch = trainer.current_epoch == trainer.max_epochs - 1
        should_compute = self.compute_during_training and (
            ((trainer.current_epoch + 1) % self.every_n_epochs == 0) or is_last_epoch
        )
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
            latent_bottlenecks, latent_topology_score = compute_topology_bottlenecks(
                self._get_normalized_original_diagrams(trainer),
                latent_diagrams,
                homology_dimensions=self.topology_dimensions,
            )
            for coeff, distances in latent_bottlenecks.items():
                for dimension, distance in distances.items():
                    pl_module.log(
                        f"latent_bottleneck_over_{coeff}_dim_{dimension}",
                        distance,
                        prog_bar=False,
                    )
            pl_module.log("latent_topology_score", latent_topology_score, prog_bar=True)
            pl_module.log(
                "latent_topology_input_metric_is_intrinsic",
                float(self.input_metric == "intrinsic_klein"),
                prog_bar=False,
            )
            self._save_pd_figures(
                trainer,
                latent_diagrams,
                wandb_key="latent_pd",
                title_prefix="Latent PD",
                filename_prefix="latent_pd",
            )
            pl_module.print(f"--- Logged normalized latent persistence diagrams. H1/H2 mean bottleneck = {latent_topology_score:.4f} ---\n")
            if is_last_epoch:
                self.final_latent_pds = latent_diagrams
                self.final_latent_bottlenecks = latent_bottlenecks
                self.final_latent_topology_score = latent_topology_score

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.global_rank != 0 or not self.save_artifacts or not self._has_eval_data(trainer):
            return
        if not self.compute_during_training:
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
        normalized_original_pds = self._get_normalized_original_diagrams(trainer)
        if self.final_reconstructed_pds is not None:
            self._save_final_pd_comparison_figure(artifact_dir, original_pds, self.final_reconstructed_pds)

        for coeff in self.field_orders:
            np.savez_compressed(artifact_dir / f"original_pd_z{coeff}.npz", *original_pds[coeff])
            np.savez_compressed(
                artifact_dir / f"normalized_original_pd_z{coeff}.npz",
                *normalized_original_pds[coeff],
            )
            if self.final_reconstructed_pds is not None:
                np.savez_compressed(artifact_dir / f"reconstructed_pd_z{coeff}.npz", *self.final_reconstructed_pds[coeff])
            if self.final_latent_pds is not None:
                np.savez_compressed(
                    artifact_dir / f"normalized_latent_pd_z{coeff}.npz",
                    *self.final_latent_pds[coeff],
                )

        wandb_logger = self._wandb_logger(trainer)
        if wandb_logger is not None:
            wandb_logger.experiment.log_artifact(
                str(artifact_dir),
                name=f"pd_results_{wandb_logger.experiment.id}",
                type="persistence_diagrams",
            )

        pl_module.print(f"Final persistence artifacts saved to {artifact_dir}.")

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self._has_eval_data(trainer, stage="test"):
            return

        empirical_variance = self._empirical_latent_variances(trainer, pl_module, stage="test")
        tiny = torch.finfo(empirical_variance.dtype).tiny
        pl_module.log("test_latent_code_variance", empirical_variance, prog_bar=False)
        pl_module.log(
            "test_log_latent_code_variance",
            empirical_variance.clamp_min(tiny).log(),
            prog_bar=False,
        )

        pl_module.print("\n--- Test Evaluation: Calculating topological metrics ---")

        if self.log_reconstruction_pd:
            log_data = self._generate_log_data(trainer, pl_module, stage="test")
            reconstructed_diagrams = log_data["reconstructed_diagrams"]
            bottlenecks = log_data["bottlenecks"]
            total_bottlenecks = log_data["total_bottlenecks"]

            for k, distances in bottlenecks.items():
                for i, distance in enumerate(distances):
                    pl_module.log(f"test_bottleneck_over_{k}_dim_{i}", float(distance), prog_bar=False)
                pl_module.log(f"test_total_bottleneck_over_{k}", total_bottlenecks[k], prog_bar=False)

            self._save_pd_figures(
                trainer,
                reconstructed_diagrams,
                wandb_key="test_reconstructed_pd",
                title_prefix="Test Reconstructed PD",
                filename_prefix="test_reconstructed_pd",
                epoch_label="Test",
            )

            totals_msg = ", ".join(f"Z/{coeff}: {total_bottlenecks[coeff]:.4f}" for coeff in self.field_orders)
            pl_module.print(f"--- Finished test reconstruction topological metrics. Total Distances = {totals_msg} ---")
            self.test_reconstructed_pds = reconstructed_diagrams
            self.test_bottlenecks = bottlenecks
            self.test_total_dist = total_bottlenecks

        latent_diagrams = self._generate_latent_diagrams(trainer, pl_module, stage="test")
        if latent_diagrams is not None:
            latent_bottlenecks, latent_topology_score = compute_topology_bottlenecks(
                self._get_normalized_original_diagrams(trainer, stage="test"),
                latent_diagrams,
                homology_dimensions=self.topology_dimensions,
            )
            for coeff, distances in latent_bottlenecks.items():
                for dimension, distance in distances.items():
                    pl_module.log(
                        f"test_latent_bottleneck_over_{coeff}_dim_{dimension}",
                        distance,
                        prog_bar=False,
                    )
            pl_module.log("test_latent_topology_score", latent_topology_score, prog_bar=True)
            self._save_pd_figures(
                trainer,
                latent_diagrams,
                wandb_key="test_latent_pd",
                title_prefix="Test Latent PD",
                filename_prefix="test_latent_pd",
                epoch_label="Test",
            )
            pl_module.print(
                f"--- Logged test normalized latent persistence diagrams. H1/H2 mean bottleneck = {latent_topology_score:.4f} ---\n"
            )
            self.test_latent_pds = latent_diagrams
            self.test_latent_bottlenecks = latent_bottlenecks
            self.test_latent_topology_score = latent_topology_score

        if trainer.global_rank == 0 and self.save_artifacts:
            artifact_dir = Path(trainer.default_root_dir) / "persistence_diagrams"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            original_pds = self._get_original_diagrams(trainer, stage="test")
            normalized_original_pds = self._get_normalized_original_diagrams(trainer, stage="test")
            if self.test_reconstructed_pds is not None:
                self._save_final_pd_comparison_figure(
                    artifact_dir,
                    original_pds,
                    self.test_reconstructed_pds,
                    filename="test_pd_comparison.png",
                    title="Test Persistence Diagram Comparison",
                )

            for coeff in self.field_orders:
                np.savez_compressed(artifact_dir / f"test_original_pd_z{coeff}.npz", *original_pds[coeff])
                np.savez_compressed(
                    artifact_dir / f"test_normalized_original_pd_z{coeff}.npz",
                    *normalized_original_pds[coeff],
                )
                if self.test_reconstructed_pds is not None:
                    np.savez_compressed(artifact_dir / f"test_reconstructed_pd_z{coeff}.npz", *self.test_reconstructed_pds[coeff])
                if self.test_latent_pds is not None:
                    np.savez_compressed(
                        artifact_dir / f"test_normalized_latent_pd_z{coeff}.npz",
                        *self.test_latent_pds[coeff],
                    )

            wandb_logger = self._wandb_logger(trainer)
            if wandb_logger is not None:
                wandb_logger.experiment.log_artifact(
                    str(artifact_dir),
                    name=f"test_pd_results_{wandb_logger.experiment.id}",
                    type="persistence_diagrams",
                )
            pl_module.print(f"Test persistence artifacts saved to {artifact_dir}.")
