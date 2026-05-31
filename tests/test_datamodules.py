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

    assert train_batch.shape == (2, 16)
    assert val_batch.shape[1:] == (16,)
    assert datamodule.data_for_pd.shape == (2, 16)
    assert torch.isfinite(train_batch).all()
    assert torch.isfinite(val_batch).all()
    assert torch.isfinite(datamodule.data_for_pd).all()

