import torch
from torch.utils.data import DataLoader, TensorDataset
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split


class CirclesDatamodule(LightningDataModule):
    def __init__(self, image_linear_pixel_size: int = 40, circle_radius=0.3, num_images=10, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.image_linear_pixel_size = image_linear_pixel_size
        self.circle_radius = circle_radius
        self.num_images = num_images

    def setup(self, stage=None):
        # num_samples_per_angle = self.num_samples_per_angle
        xs = torch.linspace(0, 1, self.image_linear_pixel_size)
        ys = torch.linspace(0, 2, 2 * self.image_linear_pixel_size)

        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")

        images = []

        for i in range(self.num_images):
            # circle center
            cc_x = 1 + 2 * torch.distributions.beta.Beta(1, 1).sample()
            cc_y = 2 + torch.distributions.beta.Beta(1, 1).sample()
            # 0,0
            gx, gy = grid_x, grid_y
            c00 = (gx - cc_x) ** 2 + (gy - cc_y) ** 2 < self.circle_radius**2
            # 0,1
            gx, gy = grid_x, grid_y + 2
            c01 = (gx - cc_x) ** 2 + (gy - cc_y) ** 2 < self.circle_radius**2
            # 0,2
            gx, gy = grid_x, grid_y + 4
            c02 = (gx - cc_x) ** 2 + (gy - cc_y) ** 2 < self.circle_radius**2
            # 1,0
            gx, gy = grid_x + 1, grid_y
            c10 = (gx - cc_x) ** 2 + (gy - cc_y) ** 2 < self.circle_radius**2
            # 1,1
            gx, gy = grid_x + 1, grid_y + 2
            c11 = (gx - cc_x) ** 2 + (gy - cc_y) ** 2 < self.circle_radius**2
            # 1,2
            gx, gy = grid_x + 1, grid_y + 4
            c12 = (gx - cc_x) ** 2 + (gy - cc_y) ** 2 < self.circle_radius**2
            # 2,0
            gx, gy = grid_x + 2, grid_y
            c20 = (gx - cc_x) ** 2 + (gy - cc_y) ** 2 < self.circle_radius**2
            # 2,1
            gx, gy = grid_x + 2, grid_y + 2
            c21 = (gx - cc_x) ** 2 + (gy - cc_y) ** 2 < self.circle_radius**2
            # 2,2
            gx, gy = grid_x + 2, grid_y + 4
            c22 = (gx - cc_x) ** 2 + (gy - cc_y) ** 2 < self.circle_radius**2

            torus = c00 + c01 + c02 + c10 + c11 + c12 + c20 + c21 + c22

            klein = torus[:, : self.image_linear_pixel_size] + torch.flip(torus[:, self.image_linear_pixel_size :], dims=(0,)).to(
                torch.float
            )

            images.append(klein.flatten())

        cirle_images = torch.stack(images)

        train_data, val_data = train_test_split(cirle_images, test_size=0.2, random_state=42)

        self.train_dataset = TensorDataset(train_data)
        self.val_dataset = TensorDataset(val_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        raise NotImplementedError("Test dataloader is not implemented.")


if __name__ == "__main__":
    module = CirclesDatamodule()
    module.setup()
    filter = module.train_dataloader().dataset[0][0]
    print(filter)  # Example usage
