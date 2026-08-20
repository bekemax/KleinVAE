import inspect

import pytest
from lightning import LightningDataModule

from scripts import evaluate_checkpoint
from scripts.evaluate_checkpoint import DATAMODULES, _datamodule_class
from src.data.circles_datamodule import CirclesDatamodule


class FakeDatamodule(LightningDataModule):
    def __init__(
        self,
        patch_stride: int = 3,
        density_k: int = 300,
        batch_size: int = 1024,
        persistence_subsample_size: int = 500,
        seed: int = 28,
        split_seed: int | None = None,
    ) -> None:
        super().__init__()


@pytest.fixture
def two_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluate_checkpoint, "DATAMODULES", (CirclesDatamodule, FakeDatamodule))


def saved_hyperparameters(datamodule_class: type) -> dict[str, object]:
    """The hyperparameters Lightning would record for a default instance of the class."""
    parameters = inspect.signature(datamodule_class).parameters
    return {name: parameter.default for name, parameter in parameters.items()}


def test_circles_hyperparameters_resolve_to_circles() -> None:
    assert _datamodule_class(saved_hyperparameters(CirclesDatamodule)) is CirclesDatamodule


def test_empty_hyperparameters_keep_the_historical_default() -> None:
    assert _datamodule_class({}) is CirclesDatamodule


def test_explicit_name_selects_a_registered_datamodule() -> None:
    assert _datamodule_class({}, "CirclesDatamodule") is CirclesDatamodule


def test_unknown_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown datamodule"):
        _datamodule_class({}, "MnistDatamodule")


def test_every_registered_datamodule_exposes_the_evaluation_contract() -> None:
    required = ("persistence_subsample_size", "seed", "split_seed")
    for candidate in DATAMODULES:
        parameters = inspect.signature(candidate).parameters
        assert all(name in parameters for name in required), candidate.__name__


@pytest.mark.usefixtures("two_registered")
def test_a_distinctive_key_picks_the_matching_datamodule() -> None:
    shared = {"seed": 28, "batch_size": 1024}
    assert evaluate_checkpoint._datamodule_class({**shared, "circle_radius": 0.3}) is CirclesDatamodule
    assert evaluate_checkpoint._datamodule_class({**shared, "density_k": 300}) is FakeDatamodule


@pytest.mark.usefixtures("two_registered")
def test_hyperparameters_shared_by_both_keep_the_historical_default() -> None:
    shared = set(inspect.signature(CirclesDatamodule).parameters) & set(inspect.signature(FakeDatamodule).parameters)
    assert evaluate_checkpoint._datamodule_class({key: None for key in shared}) is CirclesDatamodule


@pytest.mark.usefixtures("two_registered")
def test_explicit_name_overrides_inference() -> None:
    circles = saved_hyperparameters(CirclesDatamodule)
    assert evaluate_checkpoint._datamodule_class(circles, "FakeDatamodule") is FakeDatamodule
