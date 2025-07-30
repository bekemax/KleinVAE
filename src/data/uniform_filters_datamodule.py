import torch
from torch.utils.data import DataLoader, TensorDataset
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split

from src.data.components.utils import generate_klein_filter_matrix


class UniformFiltersDataModule(LightningDataModule):
    def __init__(self, filter_size: int = 3, num_samples_per_angle: int = 100, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples_per_angle = num_samples_per_angle
        self.filter_size = filter_size

    def setup(self, stage=None):
        thetas_1 = torch.rand(self.num_samples_per_angle)
        thetas_2 = torch.rand(self.num_samples_per_angle)
        self.coordinates = torch.Tensor([(t1, t2) for t1 in thetas_1 for t2 in thetas_2])
        klein_filters = torch.stack(
            [
                generate_klein_filter_matrix(t1 * 2 * torch.pi, t2 * 2 * torch.pi, size=self.filter_size, midpoint=False).flatten()
                for t1, t2 in self.coordinates
            ]
        )

        klein_filters = (klein_filters + 3.32) / 6.64  # Normalize to [0, 1]

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
    module = UniformFiltersDataModule()
    module.setup()
    filter = module.train_dataloader().dataset[0][0]
    print(filter)  # Example usage
