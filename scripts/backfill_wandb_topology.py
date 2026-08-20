"""Recompute persistence diagrams for a finished sweep and attach them to its existing wandb runs."""

import argparse
import importlib.util
import json
from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
from joblib import Parallel, delayed
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.topology_utils import (
    compute_normalized_persistence_diagrams,
    compute_pairwise_bottlenecks,
    compute_persistence_diagrams,
    compute_topology_bottlenecks,
    klein_distance_matrix,
    plot_persistence_diagram,
)

ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = ROOT / "logs" / "capacity_ablation"
SEARCH_CONFIG = ROOT / "configs" / "hparams_search" / "capacity_ablation.yaml"
FIELD_ORDERS = (2, 3)
HOMOLOGY_DIMENSIONS = (1, 2)
COVARIANCE_TYPE = "full"
SCALE_QUANTILE = 0.9
MODEL_TO_KIND = {"klein_vae": "klein", "torus_vae": "torus", "vanilla_vae": "euclidean", "vanilla_vae_fourier": "euclidean"}


def sweep_grid(config_path: Path = SEARCH_CONFIG) -> list[dict[str, Any]]:
    """Every combination the search config sweeps over, including any extra swept keys."""
    params = OmegaConf.load(config_path).hydra.sweeper.params
    swept = [key for key in params if key != "experiment"]
    axes = [[item.strip() for item in str(params[key]).split(",") if item.strip()] for key in swept]

    grid = []
    for combination in product(*axes):
        point = dict(zip(swept, combination))
        point["seed"] = int(point["seed"])
        grid.append(point)
    return grid


def run_directory(point: dict[str, Any], root: Path = SWEEP_ROOT) -> Path:
    """Hydra names the subdirectory by sorted `key=value` overrides, extra swept keys included."""
    return root / ",".join(sorted(f"{key}={value}" for key, value in point.items()))


def wandb_run_id(directory: Path) -> str | None:
    """The run id wandb encoded in its own directory name, online or offline."""
    candidates = [
        candidate
        for candidate in directory.glob("wandb/*run-*")
        if candidate.is_dir() and candidate.name.startswith(("run-", "offline-run-"))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.name.split("-")[-2]).name.rsplit("-", 1)[-1]


def best_checkpoint(directory: Path, *, selection: str = "epoch") -> Path | None:
    """The checkpoint to evaluate."""
    numbered = sorted((directory / "checkpoints").glob("epoch_*.ckpt"))
    if not numbered:
        return None
    if selection == "newest":
        return max(numbered, key=lambda path: path.stat().st_mtime)
    return numbered[-1]


def _load_evaluate_checkpoint() -> Any:
    path = Path(__file__).resolve().parent / "evaluate_checkpoint.py"
    spec = importlib.util.spec_from_file_location("evaluate_checkpoint", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def saturation_ceiling(diagram: np.ndarray) -> float:
    """Largest bottleneck from this reference: every bar matched to the diagonal."""
    bars = np.asarray(diagram)
    if bars.size == 0:
        return 0.0
    bars = bars[np.isfinite(bars[:, 1])]
    return float(((bars[:, 1] - bars[:, 0]) / 2).max()) if len(bars) else 0.0


def _save_diagram_figure(path: Path, panels: Sequence[tuple[str, list[np.ndarray]]], title: str) -> Path:
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6), squeeze=False)
    fig.suptitle(title, fontsize=15)
    for ax, (label, diagram) in zip(axes[0], panels):
        plot_persistence_diagram(diagram=diagram, title=label, ax=ax)
    plt.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def compute_run(
    point: dict[str, Any],
    *,
    persistence_sample_size: int,
    output_name: str,
    sweep_root: Path = SWEEP_ROOT,
    checkpoint_selection: str = "epoch",
) -> dict[str, Any]:
    """Recompute every diagram for one run and write them next to its checkpoints."""
    directory = run_directory(point, sweep_root)
    model_kind = MODEL_TO_KIND[point["model"]]
    seed = point["seed"]
    checkpoint_path = best_checkpoint(directory, selection=checkpoint_selection)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint in {directory / 'checkpoints'}")

    evaluate_module = _load_evaluate_checkpoint()
    torch.manual_seed(seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    module = evaluate_module._load_model(checkpoint, model_kind, COVARIANCE_TYPE)
    datamodule = evaluate_module._load_data(checkpoint, persistence_sample_size=persistence_sample_size, data_seed=seed)

    topology_input = datamodule.test_data_for_pd
    topology_parameters = datamodule.test_parameters_for_pd
    with torch.inference_mode():
        means, _ = module.model.encode(topology_input)
        reconstruction = module.reconstruct(topology_input, sample=False)

    reference_diagrams = compute_normalized_persistence_diagrams(
        klein_distance_matrix(topology_parameters),
        coeffs=FIELD_ORDERS,
        scale_quantile=SCALE_QUANTILE,
    )
    latent_diagrams = compute_normalized_persistence_diagrams(
        evaluate_module._latent_distance_matrix(module, means),
        coeffs=FIELD_ORDERS,
        scale_quantile=SCALE_QUANTILE,
    )
    latent_bottlenecks, topology_score = compute_topology_bottlenecks(
        reference_diagrams, latent_diagrams, homology_dimensions=HOMOLOGY_DIMENSIONS
    )

    input_diagrams = compute_persistence_diagrams(topology_input, coeffs=list(FIELD_ORDERS))
    reconstructed_diagrams = compute_persistence_diagrams(reconstruction, coeffs=list(FIELD_ORDERS))
    ambient_bottlenecks = compute_pairwise_bottlenecks(input_diagrams, reconstructed_diagrams)

    artifact_dir = directory / output_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for coeff in FIELD_ORDERS:
        np.savez_compressed(artifact_dir / f"reference_pd_normalized_z{coeff}.npz", *reference_diagrams[coeff])
        np.savez_compressed(artifact_dir / f"latent_pd_normalized_z{coeff}.npz", *latent_diagrams[coeff])
        np.savez_compressed(artifact_dir / f"input_pd_ambient_z{coeff}.npz", *input_diagrams[coeff])
        np.savez_compressed(artifact_dir / f"reconstructed_pd_ambient_z{coeff}.npz", *reconstructed_diagrams[coeff])

    label = f"{point['arch']}/{model_kind}/seed{seed}"
    figures = {
        "topo/latent_pd": _save_diagram_figure(
            artifact_dir / "latent_pd.png",
            [(rf"Latent PD over $\mathbb{{Z}}_{c}$", latent_diagrams[c]) for c in FIELD_ORDERS],
            f"Normalized latent persistence — {label}",
        ),
        "topo/reference_pd": _save_diagram_figure(
            artifact_dir / "reference_pd.png",
            [(rf"Reference PD over $\mathbb{{Z}}_{c}$", reference_diagrams[c]) for c in FIELD_ORDERS],
            f"Normalized reference persistence (intrinsic Klein metric) — {label}",
        ),
        "topo/reconstructed_pd": _save_diagram_figure(
            artifact_dir / "reconstructed_pd.png",
            [(rf"Reconstructed PD over $\mathbb{{Z}}_{c}$", reconstructed_diagrams[c]) for c in FIELD_ORDERS],
            f"Ambient reconstruction persistence — {label}",
        ),
        "topo/input_pd": _save_diagram_figure(
            artifact_dir / "input_pd.png",
            [(rf"Input PD over $\mathbb{{Z}}_{c}$", input_diagrams[c]) for c in FIELD_ORDERS],
            f"Ambient input persistence — {label}",
        ),
    }

    summary: dict[str, Any] = {"topo/latent_topology_score": topology_score}
    for coeff, distances in latent_bottlenecks.items():
        for dimension, distance in distances.items():
            ceiling = saturation_ceiling(reference_diagrams[coeff][dimension])
            summary[f"topo/latent_bottleneck_z{coeff}_h{dimension}"] = float(distance)
            summary[f"topo/latent_ceiling_z{coeff}_h{dimension}"] = ceiling
            # 1.0 means every reference bar hit the diagonal: the number is a data property.
            summary[f"topo/latent_saturation_z{coeff}_h{dimension}"] = float(distance) / ceiling if ceiling else float("nan")
    for coeff, distances in ambient_bottlenecks.items():
        for dimension, distance in enumerate(distances):
            summary[f"topo/bottleneck_over_{coeff}_dim_{dimension}"] = float(distance)
        summary[f"topo/total_bottleneck_over_{coeff}"] = float(np.linalg.norm(distances))

    saturations = [v for k, v in summary.items() if k.startswith("topo/latent_saturation_")]
    summary["topo/latent_saturation_mean"] = float(np.nanmean(saturations))
    summary["topo/latent_metrics_at_ceiling"] = int(sum(1 for v in saturations if v >= 1 - 1e-4))

    config = {
        "arch": point["arch"],
        "model_kind": model_kind,
        "training_seed": seed,
        "hidden_dims": list(module.model.hidden_dims),
        "latent_dim": int(module.model.latent_dim),
        "topology_sample_size": len(topology_input),
        # A flat copy makes the quotient prior filterable in the UI, like arch.
        "prior_variance": float(module.prior_variance),
        "checkpoint": checkpoint_path.name,
    }
    payload = {
        "point": point,
        "model_kind": model_kind,
        "label": label,
        "run_id": wandb_run_id(directory),
        "artifact_dir": str(artifact_dir),
        "figures": {key: str(path) for key, path in figures.items()},
        "summary": summary,
        "config": config,
    }
    (artifact_dir / "backfill.json").write_text(json.dumps(payload, indent=2))
    return payload


def upload_run(payload: dict[str, Any], *, project: str, entity: str | None, dry_run: bool) -> str:
    """Reopen the existing wandb run and attach the diagrams, images and numbers to it."""
    import wandb

    run_id = payload["run_id"]
    if run_id is None:
        return f"{payload['label']}: no wandb run id on disk, skipped"
    if dry_run:
        return f"{payload['label']}: would update {project}/{run_id}"

    run = wandb.init(
        project=project,
        entity=entity,
        id=run_id,
        resume="must",
        reinit="finish_previous",
        settings=wandb.Settings(silent=True),
    )
    try:
        run.config.update(payload["config"], allow_val_change=True)
        run.summary.update(payload["summary"])
        run.log({key: wandb.Image(path) for key, path in payload["figures"].items()})
        artifact = wandb.Artifact(name=f"pd_backfill_{run_id}", type="persistence_diagrams")
        artifact.add_dir(payload["artifact_dir"])
        run.log_artifact(artifact)
    finally:
        run.finish()
    return f"{payload['label']}: updated {project}/{run_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=5, help="Parallel workers for the diagram computation.")
    parser.add_argument("--persistence-sample-size", type=int, default=400)
    parser.add_argument("--project", default="klein-vae")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--output-name", default="persistence_diagrams_backfill")
    parser.add_argument("--sweep-root", type=Path, default=SWEEP_ROOT)
    parser.add_argument("--search-config", type=Path, default=SEARCH_CONFIG)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("epoch", "newest"),
        default="epoch",
        help="Which checkpoint to evaluate when a directory holds more than one run.",
    )
    parser.add_argument("--arch", default=None, help="Restrict to one arch from the search config grid.")
    parser.add_argument("--dry-run", action="store_true", help="Compute everything but do not touch wandb.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse diagrams already computed on disk.")
    return parser.parse_args()


def backfill(
    *,
    search_config: Path = SEARCH_CONFIG,
    sweep_root: Path = SWEEP_ROOT,
    jobs: int = 5,
    persistence_sample_size: int = 400,
    project: str = "klein-vae",
    entity: str | None = None,
    output_name: str = "persistence_diagrams_backfill",
    checkpoint_selection: str = "epoch",
    arch: str | None = None,
    dry_run: bool = False,
    skip_existing: bool = False,
) -> list[str]:
    """Attach diagrams to every run of one finished sweep; returns one status line per run."""

    grid = [point for point in sweep_grid(search_config) if arch is None or point["arch"] == arch]
    available = [point for point in grid if run_directory(point, sweep_root).exists()]
    missing = [point for point in grid if point not in available]
    for point in missing:
        print(f"missing run directory for {point}, skipped")

    pending, reused = [], []
    for point in available:
        cached = run_directory(point, sweep_root) / output_name / "backfill.json"
        if skip_existing and cached.exists():
            reused.append(json.loads(cached.read_text()))
        else:
            pending.append(point)

    print(f"computing diagrams for {len(pending)} runs on {jobs} workers ({len(reused)} reused)")
    payloads = reused + Parallel(n_jobs=jobs, verbose=10)(
        delayed(compute_run)(
            point,
            persistence_sample_size=persistence_sample_size,
            output_name=output_name,
            sweep_root=sweep_root,
            checkpoint_selection=checkpoint_selection,
        )
        for point in pending
    )

    print(f"uploading {len(payloads)} runs to wandb")
    lines = []
    for payload in sorted(payloads, key=lambda item: item["label"]):
        line = upload_run(payload, project=project, entity=entity, dry_run=dry_run)
        print(line)
        lines.append(line)
    return lines


def main() -> None:
    args = parse_args()
    backfill(
        search_config=args.search_config,
        sweep_root=args.sweep_root,
        jobs=args.jobs,
        persistence_sample_size=args.persistence_sample_size,
        project=args.project,
        entity=args.entity,
        output_name=args.output_name,
        checkpoint_selection=args.checkpoint_selection,
        arch=args.arch,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
