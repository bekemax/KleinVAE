import json
from pathlib import Path

from hydra import compose, initialize_config_dir
import torch

from scripts.evaluate_checkpoint import _architecture_from_state_dict

from scripts.run_capacity_ablation import (
    ARCHS,
    MODELS,
    encoder_spec,
    parameter_count,
    parse_args,
    run_dir,
    train_command,
    write_results,
)
from src.models.components.vae import SimpleVAE


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_named_widths_match_documented_encoders() -> None:
    assert encoder_spec(ARCHS["tiny"]) == "900 → 64 → 5"
    assert encoder_spec(ARCHS["small"]) == "900 → 256 → 64 → 5"
    assert encoder_spec(ARCHS["official"]) == "900 → 1024 → 512 → 128 → 32 → 5"
    assert parameter_count(ARCHS["tiny"]) == 116_681
    assert parameter_count(ARCHS["small"]) == 495_561
    assert parameter_count(ARCHS["official"]) == 3_035_561
    assert parameter_count(ARCHS["official"]) > 6 * parameter_count(ARCHS["small"])
    assert parameter_count(ARCHS["small"]) > 4 * parameter_count(ARCHS["tiny"])
    assert encoder_spec(ARCHS["tiny"], latent_dim=3) == "900 → 64 → 9"
    assert encoder_spec(ARCHS["tiny"], latent_dim=4) == "900 → 64 → 14"
    assert parameter_count(ARCHS["tiny"], latent_dim=3) > parameter_count(ARCHS["tiny"])


def test_checkpoint_architecture_infers_full_covariance_latent_dimension() -> None:
    for latent_dim in (2, 3, 4):
        model = SimpleVAE(900, [64], latent_dim, "full")
        state_dict = {
            f"model.{key}": value for key, value in model.state_dict().items()
        }
        input_dim, hidden_dims, inferred_dim, covariance = (
            _architecture_from_state_dict(state_dict, "full")
        )
        assert (input_dim, hidden_dims, inferred_dim, covariance) == (
            900,
            [64],
            latent_dim,
            "full",
        )


def test_capacity_ablation_composes_each_named_width() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        for arch, hidden_dims in ARCHS.items():
            cfg = compose(
                config_name="main.yaml",
                overrides=["experiment=capacity_ablation", f"arch={arch}"],
            )
            assert list(cfg.hidden_dims) == hidden_dims
            assert list(cfg.model.model.hidden_dims) == hidden_dims
            assert cfg.model.model.latent_dim == 2
            assert cfg.model.model.covariance_type == "full"
            assert cfg.seed == 28
            assert cfg.data.num_images == 100_000
            assert cfg.kl_weight == 1.0


def test_train_command_pins_stable_run_directory(tmp_path: Path) -> None:
    directory = run_dir(tmp_path, 28, "tiny", "euclidean", latent_dim=3)
    command = train_command(
        python="python",
        seed=28,
        arch="tiny",
        model_kind="euclidean",
        latent_dim=3,
        run_directory=directory,
        extra_overrides=["trainer=gpu"],
    )
    assert command[2:7] == [
        "experiment=capacity_ablation",
        "model=vanilla_vae",
        "arch=tiny",
        "seed=28",
        "model.model.latent_dim=3",
    ]
    assert directory.name == "latent_dim_3"
    assert f"hydra.run.dir={directory}" in command
    assert command[-1] == "trainer=gpu"
    assert set(MODELS) == {"klein", "torus", "euclidean"}


def test_cli_forwards_hydra_overrides_after_filters(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_capacity_ablation.py",
            "--arch",
            "tiny",
            "--model",
            "euclidean",
            "--latent-dim",
            "3",
            "--latent-dim",
            "4",
            "extras.print_config=False",
        ],
    )
    args = parse_args()
    assert args.arch == ["tiny"]
    assert args.model == ["euclidean"]
    assert args.latent_dim == [3, 4]
    assert args.overrides == ["extras.print_config=False"]


def test_write_results_metadata_uses_eval_row_sizes(tmp_path: Path) -> None:
    rows = [
        {
            "arch": "tiny",
            "model": "euclidean",
            "encoder": "900 → 64 → 5",
            "n_params": 116_681,
            "recon_nll": 1.0,
            "observations": 32,
            "topology_sample_size": 8,
        },
        {
            "arch": "tiny",
            "model": "klein",
            "encoder": "900 → 64 → 5",
            "n_params": 116_681,
            "recon_nll": 2.0,
            "observations": 32,
            "topology_sample_size": 8,
        },
    ]
    write_results(rows, tmp_path, seed=28)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["test_observations"] == 32
    assert metadata["topology_sample_size"] == 8
    assert metadata["latent_dimensions"] == [2]

    rows[1]["observations"] = 16
    write_results(rows, tmp_path, seed=28)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["test_observations"] == [32, 16]
