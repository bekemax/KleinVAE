"""Capacity ablation.

Training is a Hydra multirun over one named sweep from `configs/hparams_search`;
this script launches it and collects the runs into `results/<ablation>/results.csv`.

    uv run python scripts/run_capacity_ablation.py
    uv run python scripts/run_capacity_ablation.py --ablation latent_dim_ablation
    uv run python scripts/run_capacity_ablation.py --dry-run
    uv run python scripts/run_capacity_ablation.py --collect-only
    uv run python scripts/run_capacity_ablation.py trainer=gpu
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import rootutils
import torch
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.vae import SimpleVAE

ROOT = Path(__file__).resolve().parents[1]

_matplotlib_cache = Path("/tmp/kleinvae-matplotlib")
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))
os.environ.setdefault("MPLBACKEND", "Agg")

ARCHS: dict[str, list[int]] = {
    "tiny": [64],
    "small": [256, 64],
    "official": [1024, 512, 128, 32],
}
MODELS: dict[str, str] = {
    "klein": "klein_vae",
    "torus": "torus_vae",
    "euclidean": "vanilla_vae",
}
INPUT_DIM = 900
LATENT_DIM = 2
COVARIANCE_TYPE = "full"


def encoder_spec(hidden_dims: Sequence[int], *, latent_dim: int = LATENT_DIM) -> str:
    covariance_parameter_dim = latent_dim * (latent_dim + 1) // 2
    widths = [INPUT_DIM, *hidden_dims, latent_dim + covariance_parameter_dim]
    return " → ".join(str(width) for width in widths)


def parameter_count(hidden_dims: Sequence[int], *, latent_dim: int = LATENT_DIM) -> int:
    model = SimpleVAE(
        input_dim=INPUT_DIM,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        covariance_type=COVARIANCE_TYPE,
    )
    return sum(parameter.numel() for parameter in model.parameters())


def run_dir(root: Path, seed: int, arch: str, model_kind: str, latent_dim: int = LATENT_DIM) -> Path:
    directory = root / "logs" / "capacity_ablation" / f"seed{seed}" / arch / model_kind
    return directory if latent_dim == LATENT_DIM else directory / f"latent_dim_{latent_dim}"


def best_checkpoint(directory: Path) -> Path | None:
    checkpoint_dir = directory / "checkpoints"
    numbered = sorted(checkpoint_dir.glob("epoch_*.ckpt"))
    if numbered:
        return numbered[-1]
    last = checkpoint_dir / "last.ckpt"
    return last if last.exists() else None


def train_command(
    *,
    python: str,
    seed: int,
    arch: str,
    model_kind: str,
    latent_dim: int,
    run_directory: Path,
    extra_overrides: Sequence[str],
) -> list[str]:
    return [
        python,
        str(ROOT / "main.py"),
        "experiment=capacity_ablation",
        f"model={MODELS[model_kind]}",
        f"arch={arch}",
        f"seed={seed}",
        f"model.model.latent_dim={latent_dim}",
        f"hydra.run.dir={run_directory}",
        *extra_overrides,
    ]


def sweep_command(
    *,
    python: str,
    ablation: str,
    sweep_directory: Path,
    extra_overrides: Sequence[str],
) -> list[str]:
    """Hydra multirun for one named ablation, e.g. `hparams_search=capacity_ablation`."""

    return [
        python,
        str(ROOT / "main.py"),
        "-m",
        f"hparams_search={ablation}",
        f"hydra.sweep.dir={sweep_directory}",
        # Name subdirectories after the swept overrides so a rerun lands in the same place.
        "hydra.sweep.subdir=${hydra.job.override_dirname}",
        "hydra.job.config.override_dirname.exclude_keys=[hparams_search,experiment,logger,trainer]",
        *extra_overrides,
    ]


MODULE_TARGET_TO_MODEL_KIND = {
    "KleinVAEModule": "klein",
    "TorusVAEModule": "torus",
    "VAEModule": "euclidean",
}


def unsynced_wandb_runs(sweep_directory: Path) -> list[Path]:
    """Offline wandb runs under the sweep, minus the ones already uploaded."""
    runs = sorted(sweep_directory.glob("**/wandb/offline-run-*"))
    return [run for run in runs if run.is_dir() and not Path(f"{run}.synced").exists()]


def sync_wandb(sweep_directory: Path) -> int:
    """Upload offline runs. A sync failure must not cost the evaluation that follows."""
    runs = unsynced_wandb_runs(sweep_directory)
    if not runs:
        return 0
    print(f"syncing {len(runs)} offline wandb runs")
    result = subprocess.run(
        [sys.executable, "-m", "wandb", "sync", *(str(run) for run in runs)],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        print(f"wandb sync exited with {result.returncode}; sync manually before deleting logs")
        return len(runs)
    for run in runs:
        Path(f"{run}.synced").touch()
    return len(runs)


def upload_topology(
    *,
    ablation: str,
    sweep_directory: Path,
    persistence_sample_size: int,
    jobs: int,
) -> None:
    """Attach persistence diagrams to the sweep's wandb runs, which train without them."""
    from scripts.backfill_wandb_topology import backfill

    backfill(
        search_config=ROOT / "configs" / "hparams_search" / f"{ablation}.yaml",
        sweep_root=sweep_directory,
        jobs=jobs,
        persistence_sample_size=persistence_sample_size,
    )


def sweep_run_directories(sweep_directory: Path) -> list[Path]:
    """Every Hydra run under a sweep directory, identified by its saved config."""

    return sorted(path.parent.parent for path in sweep_directory.glob("**/.hydra/config.yaml"))


def run_metadata(run_directory: Path) -> dict[str, Any]:
    """Recover arch, model kind, seed and latent dim from the config Hydra saved for the run."""

    config = OmegaConf.load(run_directory / ".hydra" / "config.yaml")
    target = str(OmegaConf.select(config, "model._target_", default="")).rsplit(".", 1)[-1]
    if target not in MODULE_TARGET_TO_MODEL_KIND:
        raise ValueError(f"Unrecognised model target {target!r} in {run_directory}")
    return {
        "arch": str(OmegaConf.select(config, "arch", default="unknown")),
        "model_kind": MODULE_TARGET_TO_MODEL_KIND[target],
        "seed": int(OmegaConf.select(config, "seed")),
        "latent_dim": int(OmegaConf.select(config, "model.model.latent_dim", default=LATENT_DIM)),
    }


def stale_evaluation_reason(
    *,
    row: dict[str, Any] | None,
    row_path: Path | None,
    checkpoint_path: Path | None,
    persistence_sample_size: int,
) -> str | None:
    """Why a stored evaluation must be recomputed, or None when it can be reused."""

    if row is None or row_path is None:
        return "not evaluated yet"
    # Rows computed at different sample sizes are not comparable, so never mix them in one table.
    if int(row.get("topology_sample_size", -1)) != persistence_sample_size:
        return f"sample size changed {row.get('topology_sample_size')} -> {persistence_sample_size}"
    if checkpoint_path is not None and checkpoint_path.stat().st_mtime > row_path.stat().st_mtime:
        return "checkpoint is newer than the stored evaluation"
    return None


def _load_evaluate_checkpoint() -> Any:
    path = Path(__file__).resolve().parent / "evaluate_checkpoint.py"
    spec = importlib.util.spec_from_file_location("evaluate_checkpoint", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_run(
    *,
    directory: Path,
    arch: str,
    model_kind: str,
    seed: int,
    persistence_sample_size: int,
    evaluate_module: Any,
) -> dict[str, Any]:
    checkpoint_path = best_checkpoint(directory)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint in {directory / 'checkpoints'}")
    torch.manual_seed(seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    module = evaluate_module._load_model(checkpoint, model_kind, COVARIANCE_TYPE)
    datamodule = evaluate_module._load_data(
        checkpoint,
        persistence_sample_size=persistence_sample_size,
        data_seed=seed,
    )
    hidden_dims = list(module.model.hidden_dims)
    latent_dim = int(module.model.latent_dim)
    return {
        "arch": arch,
        "model": model_kind,
        "latent_dim": latent_dim,
        "hidden_dims": hidden_dims,
        "encoder": encoder_spec(hidden_dims, latent_dim=latent_dim),
        # Counted from the loaded network so a latent feature map widens the decoder here too.
        "n_params": sum(parameter.numel() for parameter in module.model.parameters()),
        "training_seed": seed,
        "evaluation_seed": seed,
        "checkpoint": str(checkpoint_path),
        "model_kind": model_kind,
        **evaluate_module.evaluate(
            module,
            datamodule,
            split="test",
            scale_quantile=0.9,
            intrinsic_input_metric=True,
        ),
    }


def _shared_field(rows: Sequence[dict[str, Any]], key: str) -> Any:
    values = [row[key] for row in rows if key in row]
    if not values:
        return None
    if all(value == values[0] for value in values):
        return values[0]
    return values


def write_results(
    rows: list[dict[str, Any]],
    output_dir: Path,
    seed: int | Sequence[int] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    default_seed = seed if isinstance(seed, int) else (list(seed)[0] if seed else 28)
    for row in rows:
        row.setdefault("latent_dim", LATENT_DIM)
        row.setdefault("training_seed", default_seed)
        row.setdefault("evaluation_seed", row["training_seed"])
        seed_val = row["training_seed"]
        raw_path = raw_dir / (f"{row['arch']}_{row['model']}_latent{row['latent_dim']}_seed{seed_val}.json")
        raw_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    table = pd.DataFrame(rows)
    preferred = [
        "arch",
        "model",
        "latent_dim",
        "training_seed",
        "evaluation_seed",
        "encoder",
        "n_params",
        "recon_nll",
        "negative_elbo",
        "latent_topology_score",
        "log_empirical_latent_code_variance",
        "kl",
        "posterior_log_variance",
    ]
    ordered = [column for column in preferred if column in table.columns]
    ordered.extend(column for column in table.columns if column not in ordered)
    table = table.loc[:, ordered]
    arch_order = {"official": 0, "small": 1, "tiny": 2}
    model_order = {"klein": 0, "torus": 1, "euclidean": 2}
    table["_arch_key"] = table["arch"].map(lambda value: arch_order.get(value, 99))
    table["_model_key"] = table["model"].map(lambda value: model_order.get(value, 99))
    table = table.sort_values(by=["_arch_key", "_model_key", "latent_dim", "training_seed"]).drop(columns=["_arch_key", "_model_key"])
    csv_path = output_dir / "results.csv"
    table.to_csv(csv_path, index=False)
    metadata = {
        "training_seeds": sorted({int(row["training_seed"]) for row in rows}),
        "evaluation_seeds": sorted({int(row["evaluation_seed"]) for row in rows}),
        "latent_dimensions": sorted({int(row["latent_dim"]) for row in rows}),
        "checkpoint_selection": "lowest validation loss",
        "architectures": {
            name: {
                "hidden_dims": hidden_dims,
                "encoder_2d": encoder_spec(hidden_dims),
                "n_params_2d": parameter_count(hidden_dims),
            }
            for name, hidden_dims in ARCHS.items()
        },
        "models": list(MODELS),
        "training_config": "configs/experiment/capacity_ablation.yaml",
        "test_observations": _shared_field(rows, "observations"),
        "topology_sample_size": _shared_field(rows, "topology_sample_size"),
        "coefficient_fields": [2, 3],
        "homology_dimensions": [1, 2],
        "distance_normalization_quantile": 0.9,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--ablation",
        default="capacity_ablation",
        help="Name of the configs/hparams_search sweep to run, e.g. capacity_ablation.",
    )
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--arch", action="append", choices=tuple(ARCHS))
    parser.add_argument("--model", action="append", choices=tuple(MODELS))
    parser.add_argument("--latent-dim", type=int, action="append")
    parser.add_argument("--dry-run", action="store_true", help="Print the train commands and exit.")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Evaluate existing checkpoints; do not train.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip training when a numbered checkpoint already exists (default: true).",
    )
    parser.add_argument(
        "--sync-wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload offline wandb runs after the sweep (default: true).",
    )
    parser.add_argument(
        "--upload-topology",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attach persistence diagrams to the sweep's wandb runs (default: true).",
    )
    parser.add_argument(
        "--topology-jobs",
        type=int,
        default=5,
        help="Parallel workers for the diagram computation.",
    )
    parser.add_argument(
        "--skip-evaluated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse a stored evaluation unless its checkpoint or sample size changed (default: true).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Tiny data and one epoch. Skips topology evaluation.",
    )
    parser.add_argument(
        "--persistence-sample-size",
        type=int,
        default=400,
        help="Official Klein-Circles evaluation uses 400.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Extra Hydra overrides forwarded to each train job, e.g. trainer=gpu.",
    )
    args = parser.parse_args()
    args.seed = args.seed or [28]
    args.arch = args.arch or list(ARCHS)
    args.model = args.model or list(MODELS)
    args.latent_dim = args.latent_dim or [LATENT_DIM]
    if any(latent_dim <= 0 for latent_dim in args.latent_dim):
        parser.error("--latent-dim must be positive")
    return args


def main() -> None:
    args = parse_args()
    extra_overrides = list(args.overrides)
    if args.fast:
        extra_overrides.extend(
            [
                "data.num_images=64",
                "data.batch_size=16",
                "data.persistence_subsample_size=8",
                "trainer.min_epochs=1",
                "trainer.max_epochs=1",
                "callbacks.topology_metrics.every_n_epochs=1",
            ]
        )

    print("arch         latent  encoder                                  params")
    for arch in args.arch:
        hidden_dims = ARCHS[arch]
        for latent_dim in args.latent_dim:
            print(
                f"{arch:<12} {latent_dim:<7} "
                f"{encoder_spec(hidden_dims, latent_dim=latent_dim):<40} "
                f"{parameter_count(hidden_dims, latent_dim=latent_dim):,}"
            )

    sweep_directory = ROOT / "logs" / args.ablation
    output_dir = ROOT / "results" / args.ablation
    command = sweep_command(
        python=sys.executable,
        ablation=args.ablation,
        sweep_directory=sweep_directory,
        extra_overrides=extra_overrides,
    )

    if args.dry_run:
        print(" ".join(command))
        return

    if not args.collect_only:
        print(f"sweep {args.ablation} -> {sweep_directory}")
        subprocess.run(command, cwd=ROOT, check=True)

    if args.sync_wandb:
        sync_wandb(sweep_directory)

    if args.fast and not args.collect_only:
        return

    evaluate_module = _load_evaluate_checkpoint()
    raw_dir = output_dir / "raw"
    existing_rows: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    existing_paths: dict[tuple[str, str, int, int], Path] = {}
    if raw_dir.exists():
        for path in raw_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                data.setdefault("latent_dim", LATENT_DIM)
                key = (
                    data["arch"],
                    data["model"],
                    int(data.get("training_seed", 28)),
                    int(data["latent_dim"]),
                )
                existing_rows[key] = data
                existing_paths[key] = path
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

    run_directories = sweep_run_directories(sweep_directory)
    if not run_directories:
        raise SystemExit(f"No Hydra runs under {sweep_directory}; run without --collect-only first")
    print(f"collecting {len(run_directories)} runs from {sweep_directory}")

    for directory in run_directories:
        metadata = run_metadata(directory)
        label = f"{metadata['arch']}/{metadata['model_kind']}/latent{metadata['latent_dim']}/seed{metadata['seed']}"
        key = (metadata["arch"], metadata["model_kind"], metadata["seed"], metadata["latent_dim"])
        if args.skip_evaluated:
            reason = stale_evaluation_reason(
                row=existing_rows.get(key),
                row_path=existing_paths.get(key),
                checkpoint_path=best_checkpoint(directory),
                persistence_sample_size=args.persistence_sample_size,
            )
            if reason is None:
                print(f"skip eval {label}: already evaluated")
                continue
            if key in existing_rows:
                print(f"re-evaluating {label}: {reason}")
        try:
            row = evaluate_run(
                directory=directory,
                arch=metadata["arch"],
                model_kind=metadata["model_kind"],
                seed=metadata["seed"],
                persistence_sample_size=args.persistence_sample_size,
                evaluate_module=evaluate_module,
            )
        except FileNotFoundError as error:
            print(f"skip eval {label}: {error}")
            continue
        existing_rows[key] = row
        print(f"{label}: recon_nll={row['recon_nll']:.4f}  topo={row['latent_topology_score']:.4f}  n_params={row['n_params']:,}")

    rows = list(existing_rows.values())
    if not rows:
        raise SystemExit("No checkpoints to evaluate")
    csv_path = write_results(rows, output_dir, args.seed)
    print(csv_path)

    if args.upload_topology and not args.fast:
        # A wandb failure here must not cost the results table that is already on disk.
        try:
            upload_topology(
                ablation=args.ablation,
                sweep_directory=sweep_directory,
                persistence_sample_size=args.persistence_sample_size,
                jobs=args.topology_jobs,
            )
        except Exception as error:
            print(f"topology upload failed ({error}); rerun scripts/backfill_wandb_topology.py")


if __name__ == "__main__":
    main()
