"""Capacity ablation: Klein-Circles protocol at three MLP widths.

The official encoder ``900 → 1024 → 512 → 128 → 32 → 5`` is ~3.0M parameters
on a 2-parameter binary-disk dataset. This sweep holds data, seed, optimizer,
latent dimension, and losses fixed and varies ``hidden_dims``.

    uv run python scripts/run_capacity_ablation.py
    uv run python scripts/run_capacity_ablation.py --dry-run
    uv run python scripts/run_capacity_ablation.py --collect-only
    uv run python scripts/run_capacity_ablation.py --arch tiny --model euclidean
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
import torch

import rootutils

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
        "n_params": parameter_count(hidden_dims, latent_dim=latent_dim),
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

    cells = [
        (arch, model_kind, seed, latent_dim)
        for arch in args.arch
        for model_kind in args.model
        for seed in args.seed
        for latent_dim in args.latent_dim
        if model_kind == "euclidean" or latent_dim == LATENT_DIM
    ]
    if not cells:
        raise SystemExit("No valid cells: Klein and Torus require latent_dim=2")
    print(f"grid: {len(cells)} runs  (arch × model × seed × valid latent dim)")

    if not args.collect_only:
        for arch, model_kind, seed, latent_dim in cells:
            directory = run_dir(ROOT, seed, arch, model_kind, latent_dim)
            command = train_command(
                python=sys.executable,
                seed=seed,
                arch=arch,
                model_kind=model_kind,
                latent_dim=latent_dim,
                run_directory=directory,
                extra_overrides=extra_overrides,
            )
            existing = best_checkpoint(directory)
            label = f"{arch}/{model_kind}/latent{latent_dim}/seed{seed}"
            if args.dry_run:
                skipped = "  # skip-existing" if args.skip_existing and existing is not None else ""
                print(" ".join(command) + skipped)
                continue
            if args.skip_existing and existing is not None:
                print(f"skip train {label}: {existing}")
                continue
            directory.mkdir(parents=True, exist_ok=True)
            print(f"train {label}")
            subprocess.run(command, cwd=ROOT, check=True)

    if args.dry_run or (args.fast and not args.collect_only):
        return

    evaluate_module = _load_evaluate_checkpoint()
    raw_dir = ROOT / "results" / "capacity_ablation" / "raw"
    existing_rows: dict[tuple[str, str, int, int], dict[str, Any]] = {}
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
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

    for arch, model_kind, seed, latent_dim in cells:
        directory = run_dir(ROOT, seed, arch, model_kind, latent_dim)
        label = f"{arch}/{model_kind}/latent{latent_dim}/seed{seed}"
        try:
            row = evaluate_run(
                directory=directory,
                arch=arch,
                model_kind=model_kind,
                seed=seed,
                persistence_sample_size=args.persistence_sample_size,
                evaluate_module=evaluate_module,
            )
        except FileNotFoundError as error:
            print(f"skip eval {label}: {error}")
            continue
        existing_rows[(arch, model_kind, seed, latent_dim)] = row
        print(f"{label}: recon_nll={row['recon_nll']:.4f}  topo={row['latent_topology_score']:.4f}  n_params={row['n_params']:,}")

    rows = list(existing_rows.values())
    if not rows:
        raise SystemExit("No checkpoints to evaluate")
    csv_path = write_results(rows, ROOT / "results" / "capacity_ablation", args.seed)
    print(csv_path)


if __name__ == "__main__":
    main()
