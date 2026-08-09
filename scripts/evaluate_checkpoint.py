"""Evaluate reconstruction and quotient-aware latent topology from a checkpoint.

The topology score compares H1 and H2 persistence diagrams over Z2 and Z3 on
the same held-out observations.  Input and latent pairwise metrics are each
normalized by their 90th-percentile nonzero distance before persistent
homology, avoiding a meaningless comparison between pixel and latent units.
"""

import argparse
import inspect
import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import rootutils
import torch
from torch import nn

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.circles_datamodule import CirclesDatamodule
from src.models.components.vae import SimpleVAE
from src.models.klein_vae_module import KleinVAEModule
from src.models.torus_vae_module import TorusVAEModule
from src.models.vae_module import VAEModule
from src.utils.topology_utils import (
    compute_normalized_persistence_diagrams,
    compute_topology_bottlenecks,
    euclidean_distance_matrix,
    klein_distance_matrix,
    project_to_klein,
    project_to_torus,
    torus_distance_matrix,
)


def _architecture_from_state_dict(state_dict: dict[str, torch.Tensor], covariance_type: str | None) -> tuple[int, list[int], int, str]:
    encoder_weights = sorted(
        (
            (int(key.split(".")[2]), value)
            for key, value in state_dict.items()
            if key.startswith("model.encoder.") and key.endswith(".weight")
        ),
        key=lambda item: item[0],
    )
    if not encoder_weights:
        raise ValueError("Checkpoint contains no SimpleVAE encoder weights")

    input_dim = int(encoder_weights[0][1].shape[1])
    hidden_dims = [int(weight.shape[0]) for _, weight in encoder_weights[:-1]]
    output_dim = int(encoder_weights[-1][1].shape[0])
    latent_dim = 2
    covariance_parameter_dim = output_dim - latent_dim
    if covariance_type is None:
        if covariance_parameter_dim == latent_dim * (latent_dim + 1) // 2:
            covariance_type = "full"
        elif covariance_parameter_dim == latent_dim:
            covariance_type = "diagonal"
        elif covariance_parameter_dim == 1:
            raise ValueError("Scalar covariance is ambiguous; pass --covariance-type isotropic")
        else:
            raise ValueError("Cannot infer covariance type from checkpoint")
    return input_dim, hidden_dims, latent_dim, covariance_type


def _load_model(checkpoint: dict[str, Any], model_kind: str, covariance_type: str | None) -> VAEModule:
    input_dim, hidden_dims, latent_dim, inferred_covariance = _architecture_from_state_dict(checkpoint["state_dict"], covariance_type)
    network = SimpleVAE(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        covariance_type=inferred_covariance,  # type: ignore[arg-type]
    )
    hparams = checkpoint["hyper_parameters"]
    common = {
        "model": network,
        "optimizer": partial(torch.optim.Adam, lr=float(hparams.get("lr", 1e-3))),
        "scheduler": None,
        "recon_loss": nn.BCELoss(reduction="sum"),
        "prior_mean": float(hparams.get("prior_mean", 0.0)),
        "prior_variance": float(hparams.get("prior_variance", 1.0)),
        "lr": float(hparams.get("lr", 1e-3)),
        "kl_weight": float(hparams.get("kl_weight", 1.0)),
    }
    if model_kind == "klein":
        module: VAEModule = KleinVAEModule(**common)
    elif model_kind == "torus":
        module = TorusVAEModule(**common)
    elif model_kind == "euclidean":
        module = VAEModule(**common)
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(model_kind)

    module.load_state_dict(checkpoint["state_dict"], strict=True)
    module.eval()
    return module


def _load_data(
    checkpoint: dict[str, Any],
    *,
    persistence_sample_size: int,
    data_seed: int | None,
) -> CirclesDatamodule:
    saved = dict(checkpoint.get("datamodule_hyper_parameters", {}))
    allowed = set(inspect.signature(CirclesDatamodule).parameters)
    kwargs = {key: value for key, value in saved.items() if key in allowed}
    kwargs["persistence_subsample_size"] = persistence_sample_size
    if data_seed is not None:
        kwargs["seed"] = data_seed
        kwargs["split_seed"] = None
    datamodule = CirclesDatamodule(**kwargs)
    datamodule.setup()
    return datamodule


def _latent_distance_matrix(module: VAEModule, means: torch.Tensor) -> torch.Tensor:
    if isinstance(module, KleinVAEModule):
        return klein_distance_matrix(project_to_klein(means))
    if isinstance(module, TorusVAEModule):
        return torus_distance_matrix(project_to_torus(means, stack=True))
    return euclidean_distance_matrix(means)


def evaluate(
    module: VAEModule,
    datamodule: CirclesDatamodule,
    *,
    split: str,
    scale_quantile: float,
    intrinsic_input_metric: bool,
) -> dict[str, Any]:
    if split == "validation":
        dataset = datamodule.val_dataset
        topology_input = datamodule.validation_data_for_pd
        topology_parameters = getattr(datamodule, "validation_parameters_for_pd", None)
    else:
        dataset = datamodule.test_dataset
        topology_input = datamodule.test_data_for_pd
        topology_parameters = getattr(datamodule, "test_parameters_for_pd", None)
    if dataset is None or topology_input is None or len(dataset) == 0 or len(topology_input) == 0:
        raise ValueError(f"The requested {split} split is empty")

    totals = {
        "recon_nll": 0.0,
        "elbo_recon_nll": 0.0,
        "kl": 0.0,
        "posterior_log_variance": 0.0,
    }
    observation_count = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)
    with torch.inference_mode():
        for (x,) in loader:
            means, covariance_parameters = module.model.encode(x)
            mean_reconstruction = module.model.decode(module.project_latent(means))
            sampled_codes = module.model.reparameterize(means, covariance_parameters)
            sampled_reconstruction = module.model.decode(module.project_latent(sampled_codes))
            batch_size = x.shape[0]
            totals["recon_nll"] += float(module.reconstruction_nll(x, mean_reconstruction)) * batch_size
            totals["elbo_recon_nll"] += float(module.reconstruction_nll(x, sampled_reconstruction)) * batch_size
            totals["kl"] += float(module.kl_loss(means, covariance_parameters)) * batch_size
            totals["posterior_log_variance"] += float(module.model.posterior_log_variance(covariance_parameters).mean()) * batch_size
            observation_count += batch_size

        means, covariance_parameters = module.model.encode(topology_input)
        sampled_codes = module.model.reparameterize(means, covariance_parameters)
        sampled_codes = module.project_latent(sampled_codes)
        empirical_latent_code_variance = sampled_codes.var(dim=0, unbiased=True).sum()

    if intrinsic_input_metric:
        if topology_parameters is None:
            raise ValueError("The datamodule does not expose intrinsic Klein parameters")
        input_distance_matrix = klein_distance_matrix(topology_parameters)
    else:
        input_distance_matrix = euclidean_distance_matrix(topology_input)
    input_diagrams = compute_normalized_persistence_diagrams(
        input_distance_matrix,
        coeffs=(2, 3),
        scale_quantile=scale_quantile,
    )
    latent_diagrams = compute_normalized_persistence_diagrams(
        _latent_distance_matrix(module, means),
        coeffs=(2, 3),
        scale_quantile=scale_quantile,
    )
    latent_bottlenecks, topology_score = compute_topology_bottlenecks(
        input_diagrams,
        latent_diagrams,
        homology_dimensions=(1, 2),
    )

    recon_nll = totals["recon_nll"] / observation_count
    elbo_recon_nll = totals["elbo_recon_nll"] / observation_count
    kl = totals["kl"] / observation_count
    posterior_log_variance = totals["posterior_log_variance"] / observation_count
    empirical_latent_code_variance_value = float(empirical_latent_code_variance)
    result: dict[str, Any] = {
        "split": split,
        "observations": observation_count,
        "topology_sample_size": len(topology_input),
        "scale_quantile": scale_quantile,
        "input_metric": "intrinsic_klein" if intrinsic_input_metric else "ambient_euclidean",
        "reconstruction_mode": "posterior_mean",
        "recon_nll": recon_nll,
        "elbo_recon_nll": elbo_recon_nll,
        "kl": kl,
        "negative_elbo": elbo_recon_nll + kl,
        "posterior_log_variance": posterior_log_variance,
        "empirical_latent_code_variance": empirical_latent_code_variance_value,
        "log_empirical_latent_code_variance": float(np.log(max(empirical_latent_code_variance_value, np.finfo(float).tiny))),
        "latent_topology_score": topology_score,
    }
    for coeff, dimensions in latent_bottlenecks.items():
        for dimension, distance in dimensions.items():
            result[f"latent_bottleneck_z{coeff}_h{dimension}"] = distance
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--model-kind", choices=("klein", "torus", "euclidean"), required=True)
    parser.add_argument(
        "--covariance-type",
        choices=("isotropic", "diagonal", "full"),
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--persistence-sample-size", type=int, default=500)
    parser.add_argument(
        "--data-seed",
        type=int,
        help="Override the checkpoint's data seed for an independent holdout dataset.",
    )
    parser.add_argument("--scale-quantile", type=float, default=0.9)
    parser.add_argument(
        "--intrinsic-input-metric",
        action="store_true",
        help="Use known generating Klein coordinates for the input PD (synthetic data only).",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.data_seed if args.data_seed is not None else 28)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    module = _load_model(checkpoint, args.model_kind, args.covariance_type)
    datamodule = _load_data(
        checkpoint,
        persistence_sample_size=args.persistence_sample_size,
        data_seed=args.data_seed,
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "model_kind": args.model_kind,
        **evaluate(
            module,
            datamodule,
            split=args.split,
            scale_quantile=args.scale_quantile,
            intrinsic_input_metric=args.intrinsic_input_metric,
        ),
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
