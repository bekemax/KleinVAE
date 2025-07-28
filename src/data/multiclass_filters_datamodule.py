import torch
from torch.utils.data import DataLoader, TensorDataset
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split

from src.data.components.utils import generate_klein_filter_matrix

from typing import List, Tuple


class MulticlassFiltersDataModule(LightningDataModule):
    def __init__(
        self,
        num_classes: int = 3,
        filter_size: int = 3,
        num_samples_per_angle: int = 100,
        batch_size: int = 32,
        class_centers: List[Tuple[float, float]] = [(0.0, 0.0), (0.7, 0), (0.3, 1)],
        class_std: float = 0.1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples_per_angle = num_samples_per_angle
        self.filter_size = filter_size
        self.num_classes = num_classes
        self.class_centers = class_centers
        self.class_std = class_std

    def setup(self, stage=None):
        theta1_samples_list = []
        theta2_samples_list = []
        for center in self.class_centers:
            t1 = torch.normal(mean=center[0], std=self.class_std, size=(self.num_samples_per_angle,))
            t2 = torch.normal(mean=center[1], std=self.class_std, size=(self.num_samples_per_angle,))
            t1 = t1.clamp(0.0, torch.pi)
            t2 = t2.clamp(0.0, 2 * torch.pi)
            theta1_samples_list.append(t1)
            theta2_samples_list.append(t2)
        thetas_1 = torch.cat(theta1_samples_list)
        thetas_2 = torch.cat(theta2_samples_list)
        klein_filters = torch.stack(
            [generate_klein_filter_matrix(t1, t2, size=self.filter_size, midpoint=False) for t1, t2 in zip(thetas_1, thetas_2)]
        )
        thetas_1 = torch.rand(self.num_samples_per_angle) * torch.pi
        thetas_2 = torch.rand(self.num_samples_per_angle) * 2 * torch.pi
        klein_filters = torch.stack(
            [generate_klein_filter_matrix(t1, t2, size=self.filter_size, midpoint=False) for t1 in thetas_1 for t2 in thetas_2]
        )

        train_data, val_data = train_test_split(klein_filters, test_size=0.2, random_state=42)

        self.train_dataset = TensorDataset(train_data)
        self.val_dataset = TensorDataset(val_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        raise NotImplementedError("Test dataloader is not implemented.")


if __name__ == "__main__":
    module = MulticlassFiltersDataModule()
    module.setup()
    print(module.train_dataloader().dataset[0][0])  # Example usage
