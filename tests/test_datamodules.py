import torch

from src.data.circles_datamodule import CirclesDatamodule
from src.data.uniform_filters_datamodule import UniformFiltersDataModule


def test_uniform_filters_datamodule_setup_and_loaders_are_finite() -> None:
    datamodule = UniformFiltersDataModule(filter_size=3, num_samples_per_angle=3, batch_size=2)

    datamodule.setup()
    train_batch = next(iter(datamodule.train_dataloader()))[0]
    val_batch = next(iter(datamodule.val_dataloader()))[0]

    assert train_batch.shape == (2, 9)
    assert val_batch.shape[1:] == (9,)
    assert torch.isfinite(train_batch).all()
    assert torch.isfinite(val_batch).all()


def test_circles_datamodule_setup_loaders_and_persistence_data_are_finite() -> None:
    datamodule = CirclesDatamodule(
        image_linear_pixel_size=4,
        circle_radius=0.3,
        num_images=8,
        batch_size=2,
        persistence_subsample_size=2,
    )

    datamodule.setup()
    train_batch = next(iter(datamodule.train_dataloader()))[0]
    val_batch = next(iter(datamodule.val_dataloader()))[0]
    test_batch = next(iter(datamodule.test_dataloader()))[0]

    assert train_batch.shape == (2, 16)
    assert val_batch.shape[1:] == (16,)
    assert test_batch.shape[1:] == (16,)
    assert datamodule.data_for_pd.shape == (1, 16)
    assert datamodule.validation_data_for_pd.shape == (1, 16)
    assert datamodule.test_data_for_pd.shape == (1, 16)
    assert datamodule.validation_parameters_for_pd.shape == (1, 2)
    assert datamodule.test_parameters_for_pd.shape == (1, 2)
    torch.testing.assert_close(
        datamodule.data_for_pd, datamodule.validation_data_for_pd
    )
    assert torch.isfinite(train_batch).all()
    assert torch.isfinite(val_batch).all()
    assert torch.isfinite(test_batch).all()
    assert torch.isfinite(datamodule.data_for_pd).all()


def test_circles_data_generation_and_splits_are_reproducible() -> None:
    kwargs = dict(
        image_linear_pixel_size=6,
        circle_radius=0.3,
        num_images=20,
        batch_size=4,
        persistence_subsample_size=3,
        seed=28,
    )
    first = CirclesDatamodule(**kwargs)
    second = CirclesDatamodule(**kwargs)

    first.setup()
    second.setup()

    torch.testing.assert_close(first.train_dataset.tensors[0], second.train_dataset.tensors[0])
    torch.testing.assert_close(first.val_dataset.tensors[0], second.val_dataset.tensors[0])
    torch.testing.assert_close(first.test_dataset.tensors[0], second.test_dataset.tensors[0])
    torch.testing.assert_close(first.data_for_pd, second.data_for_pd)
