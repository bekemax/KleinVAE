from argparse import Namespace

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset


class CirclesDatamodule(LightningDataModule):
    """Generate the synthetic Klein-Circles dataset used in the manuscript.

    Circle centers are sampled uniformly on the ``[0, 2) x [0, 1)`` double
    cover and folded into a ``[0, 1] x [0, 1]`` Klein fundamental domain. Data
    generation and all splits use local, recorded random-number generators.
    """

    hparams: Namespace

    def __init__(
        self,
        image_linear_pixel_size: int = 30,
        circle_radius: float = 0.3,
        num_images: int = 100_000,
        batch_size: int = 1024,
        persistence_subsample_size: int = 500,
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        seed: int = 28,
        split_seed: int | None = None,
        shuffle_train: bool = True,
        generation_batch_size: int = 2048,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        if image_linear_pixel_size <= 1:
            raise ValueError("image_linear_pixel_size must be greater than one")
        if circle_radius <= 0:
            raise ValueError("circle_radius must be positive")
        if num_images < 3:
            raise ValueError("num_images must be at least three")
        if not 0 < train_fraction < 1:
            raise ValueError("train_fraction must be between zero and one")
        if not 0 < val_fraction < 1:
            raise ValueError("val_fraction must be between zero and one")
        if train_fraction + val_fraction > 1:
            raise ValueError("train_fraction + val_fraction must not exceed one")
        if persistence_subsample_size <= 0:
            raise ValueError("persistence_subsample_size must be positive")
        if generation_batch_size <= 0:
            raise ValueError("generation_batch_size must be positive")
        self.save_hyperparameters()
        self.train_dataset: TensorDataset | None = None
        self.val_dataset: TensorDataset | None = None
        self.test_dataset: TensorDataset | None = None
        self.validation_data_for_pd: torch.Tensor | None = None
        self.test_data_for_pd: torch.Tensor | None = None
        self.validation_parameters_for_pd: torch.Tensor | None = None
        self.test_parameters_for_pd: torch.Tensor | None = None
        self._generated_klein_parameters: torch.Tensor | None = None
        self.data_for_pd: torch.Tensor | None = None

    def _generate_images(self) -> torch.Tensor:
        pixel_count = self.hparams.image_linear_pixel_size
        generator = torch.Generator().manual_seed(self.hparams.seed)
        centers = torch.rand(self.hparams.num_images, 2, generator=generator)
        center_x = 1 + 2 * centers[:, 0]
        center_y = 2 + centers[:, 1]
        # The image construction has a unit-periodic x direction and a
        # y-boundary crossing that flips x.  Swapping those roles gives the
        # (u, v) convention used by project_to_klein: u is the twisted
        # direction and v is the unit-periodic direction.
        self._generated_klein_parameters = torch.stack([centers[:, 1], torch.remainder(2 * centers[:, 0], 1.0)], dim=-1)

        xs = torch.linspace(0, 1, pixel_count)
        ys = torch.linspace(0, 2, 2 * pixel_count)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        radius_squared = self.hparams.circle_radius**2

        image_batches: list[torch.Tensor] = []
        for start in range(0, self.hparams.num_images, self.hparams.generation_batch_size):
            stop = min(start + self.hparams.generation_batch_size, self.hparams.num_images)
            batch_center_x = center_x[start:stop, None, None]
            batch_center_y = center_y[start:stop, None, None]
            tiled_circle = torch.zeros(
                stop - start,
                pixel_count,
                2 * pixel_count,
                dtype=torch.bool,
            )
            for x_shift in (0.0, 1.0, 2.0):
                squared_x_distance = (grid_x + x_shift - batch_center_x).square()
                for y_shift in (0.0, 2.0, 4.0):
                    squared_distance = squared_x_distance + (grid_y + y_shift - batch_center_y).square()
                    tiled_circle.logical_or_(squared_distance < radius_squared)

            lower_sheet = tiled_circle[..., :pixel_count]
            upper_sheet = torch.flip(tiled_circle[..., pixel_count:], dims=(-2,))
            klein_image = torch.logical_or(lower_sheet, upper_sheet)
            image_batches.append(klein_image.flatten(start_dim=1).float())

        return torch.cat(image_batches, dim=0)

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return

        images = self._generate_images()
        assert self._generated_klein_parameters is not None
        parameters = self._generated_klein_parameters
        split_seed = self.hparams.seed + 1 if self.hparams.split_seed is None else self.hparams.split_seed
        split_generator = torch.Generator().manual_seed(split_seed)
        indices = torch.randperm(len(images), generator=split_generator)

        train_size = max(1, round(len(images) * self.hparams.train_fraction))
        val_size = max(1, round(len(images) * self.hparams.val_fraction))
        if train_size + val_size > len(images):
            val_size = len(images) - train_size

        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        train_data = images[train_indices]
        val_data = images[val_indices]
        test_data = images[test_indices]
        val_parameters = parameters[val_indices]
        test_parameters = parameters[test_indices]
        del images

        validation_pd_generator = torch.Generator().manual_seed(self.hparams.seed + 2)
        validation_pd_size = min(self.hparams.persistence_subsample_size, len(val_data))
        validation_pd_indices = torch.randperm(len(val_data), generator=validation_pd_generator)[:validation_pd_size]

        test_pd_generator = torch.Generator().manual_seed(self.hparams.seed + 4)
        test_pd_size = min(self.hparams.persistence_subsample_size, len(test_data))
        test_pd_indices = torch.randperm(len(test_data), generator=test_pd_generator)[:test_pd_size]

        self.train_dataset = TensorDataset(train_data)
        self.val_dataset = TensorDataset(val_data)
        self.test_dataset = TensorDataset(test_data)
        self.validation_data_for_pd = val_data[validation_pd_indices]
        self.test_data_for_pd = test_data[test_pd_indices]
        self.validation_parameters_for_pd = val_parameters[validation_pd_indices]
        self.test_parameters_for_pd = test_parameters[test_pd_indices]
        self.data_for_pd = self.validation_data_for_pd

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        loader_generator = torch.Generator().manual_seed(self.hparams.seed + 3)
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle_train,
            generator=loader_generator,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
