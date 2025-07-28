import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split


class DWPDataModule(LightningDataModule):
    def __init__(self, data_dir="data/dwp-data/conv5x5", batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        train_path = os.path.join(self.data_dir, "train.npy")
        test_path = os.path.join(self.data_dir, "test.npy")

        print(f"Loading training data from {train_path}")
        train_data = np.load(train_path)
        test_data = np.load(test_path)

        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

        self.train_dataset = TensorDataset(torch.from_numpy(train_data))
        self.val_dataset = TensorDataset(torch.from_numpy(val_data))
        self.test_dataset = TensorDataset(torch.from_numpy(test_data))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    module = DWPDataModule()
    module.setup()
    print(module)
