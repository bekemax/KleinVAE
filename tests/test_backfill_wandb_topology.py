import os
from pathlib import Path

import numpy as np

from scripts.backfill_wandb_topology import (
    best_checkpoint,
    run_directory,
    saturation_ceiling,
    sweep_grid,
    wandb_run_id,
)


def test_sweep_grid_covers_the_search_config_product() -> None:
    grid = sweep_grid()
    assert len(grid) == 27
    assert {point["arch"] for point in grid} == {"tiny", "small", "official"}
    assert {point["model"] for point in grid} == {"klein_vae", "torus_vae", "vanilla_vae"}
    assert {point["seed"] for point in grid} == {27, 28, 29}


def test_run_directory_matches_the_hydra_override_dirname(tmp_path: Path) -> None:
    point = {"arch": "official", "model": "klein_vae", "seed": 27}
    assert run_directory(point, tmp_path).name == "arch=official,model=klein_vae,seed=27"


def test_wandb_run_id_reads_offline_and_online_directories(tmp_path: Path) -> None:
    offline = tmp_path / "offline" / "wandb" / "offline-run-20260819_034201-ccxxmncw"
    offline.mkdir(parents=True)
    assert wandb_run_id(tmp_path / "offline") == "ccxxmncw"

    online = tmp_path / "online" / "wandb" / "run-20260819_020119-irxb4c8i"
    online.mkdir(parents=True)
    assert wandb_run_id(tmp_path / "online") == "irxb4c8i"


def test_wandb_run_id_is_none_without_a_run(tmp_path: Path) -> None:
    (tmp_path / "wandb").mkdir()
    assert wandb_run_id(tmp_path) is None


def test_saturation_ceiling_is_half_the_longest_bar() -> None:
    diagram = np.array([[0.0, 0.4], [0.1, 0.3], [0.2, 1.4]])
    assert saturation_ceiling(diagram) == 0.6


def test_saturation_ceiling_ignores_infinite_bars() -> None:
    diagram = np.array([[0.0, 0.5], [0.0, np.inf]])
    assert saturation_ceiling(diagram) == 0.25


def test_saturation_ceiling_of_an_empty_diagram_is_zero() -> None:
    assert saturation_ceiling(np.empty((0, 2))) == 0.0


def test_wandb_run_id_prefers_the_newest_run_of_a_rerun(tmp_path: Path) -> None:
    wandb_dir = tmp_path / "wandb"
    (wandb_dir / "offline-run-20260819_153501-oldrunid").mkdir(parents=True)
    (wandb_dir / "offline-run-20260819_163317-newrunid").mkdir(parents=True)
    assert wandb_run_id(tmp_path) == "newrunid"


def test_best_checkpoint_selects_by_epoch_or_by_write_time(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    older_but_longer = checkpoints / "epoch_292.ckpt"
    newer_but_shorter = checkpoints / "epoch_168.ckpt"
    older_but_longer.touch()
    os.utime(older_but_longer, (1_000, 1_000))
    newer_but_shorter.touch()
    os.utime(newer_but_shorter, (2_000, 2_000))

    assert best_checkpoint(tmp_path) == older_but_longer
    assert best_checkpoint(tmp_path, selection="newest") == newer_but_shorter
