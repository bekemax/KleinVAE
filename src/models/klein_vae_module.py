from src.utils import project_to_klein

import lightning as pl
import torch
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Uniform

from typing import Tuple, Union


def generate_klein_filter_matrix_batched(theta_1: torch.Tensor, theta_2: torch.Tensor, size: int = 3) -> torch.Tensor:
    """
    Batched version of generate_klein_filter_matrix.

    Args:
        theta_1: [B] tensor
        theta_2: [B] tensor
        size: Output matrix size per sample

    Returns:
        [B, size, size] tensor
    """
    B = theta_1.shape[0]
    device = theta_1.device

    coords = torch.linspace(-1, 1, size, device=device)
    X, Y = torch.meshgrid(coords, coords, indexing="ij")  # [size, size]

    # Expand to [B, size, size]
    X = X.unsqueeze(0).expand(B, -1, -1)  # [B, size, size]
    Y = Y.unsqueeze(0).expand(B, -1, -1)

    # Reshape theta to broadcast: [B, 1, 1]
    theta_1 = theta_1.view(B, 1, 1)
    theta_2 = theta_2.view(B, 1, 1)

    # Projection t = x * cos(θ₁) + y * sin(θ₁)
    t = X * torch.cos(theta_1) + Y * torch.sin(theta_1)  # [B, size, size]

    # Chebyshev 2nd poly
    cheb = 2 * t.pow(2) - 1

    # Klein filter: sin(θ₂) * t + cos(θ₂) * chebyshev(t)
    result = torch.sin(theta_2) * t + torch.cos(theta_2) * cheb  # [B, size, size]

    return result


class KleinVAEModule(pl.LightningModule):
    def __init__(self, model, sigma2: float = 5, lr: float = 1e-3, batch_size: int = 1, kl_weight: float = 1e-1):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.kl_weight = kl_weight
        self.sigma2 = sigma2

    def sample(self, num_samples, return_unprojected: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        prior_dist = Uniform(low=0, high=1)
        samples = prior_dist.sample([num_samples, 2])
        if return_unprojected:
            return project_to_klein(samples), samples
        else:
            return project_to_klein(samples)

    def reparameterize(self, mu, log_sigma, non_diag) -> Tuple[torch.Tensor, torch.Tensor]:
        var = torch.exp(log_sigma)
        batch_size = mu.shape[0]

        L = torch.zeros(batch_size, 2, 2, device=mu.device)
        L[:, 0, 0] = 1e-3  # var[:, 0]
        L[:, 1, 1] = 1e-3  # var[:, 1]
        L[:, 1, 0] = 0  # non_diag[:, 0]

        standard_normal = MultivariateNormal(torch.zeros(2), torch.eye(2))
        eps = standard_normal.sample([batch_size])  # -> shape (batch_size, 2)
        # torch.randn_like(mu) * torch.sqrt(torch.tensor(self.sigma2, device=mu.device))
        return mu + torch.matmul(L, eps.unsqueeze(-1)).squeeze(-1), L

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        theta = z * 2 * torch.pi  # z is [B, 2]
        theta_1 = theta[:, 0]
        theta_2 = theta[:, 1]

        recon_x = generate_klein_filter_matrix_batched(theta_1, theta_2, size=3)  # [B, 3, 3]
        recon_x = recon_x.view(z.shape[0], -1)  # Flatten to [B, 9]
        recon_x = (recon_x + 3.32) / 6.64  # [B, 9]

        # recon_x = generate_klein_filter_matrix(z[..., 0] * 2 * torch.pi, z[..., 1] * 2 * torch.pi, size=3).flatten().unsqueeze(0)
        # recon_x = F.sigmoid(recon_x)
        # recon_x = (recon_x + 3.32) / 6.64  # Normalize to [0, 1]

        return recon_x

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_sigma, non_diag = self.encode(x)
        z_on_plane, L = self.reparameterize(mu, log_sigma, non_diag)
        z_on_klein = project_to_klein(z_on_plane)

        recon_x = self.decode(z_on_klein)

        return recon_x, mu, L

    def _vae_loss(self, x, recon_x, mu, L) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="mean")

        # Create multivariate normal with batched L matrices
        q = MultivariateNormal(mu, scale_tril=L)

        prior_loc = torch.zeros_like(mu)
        prior_scale = torch.eye(2, device=mu.device).unsqueeze(0).repeat(mu.size(0), 1, 1) * self.sigma2
        prior_dist = MultivariateNormal(prior_loc, scale_tril=prior_scale)

        kl_div = kl_divergence(q, prior_dist).mean()
        return recon_loss + self.kl_weight * kl_div, recon_loss, kl_div

    def training_step(self, batch, batch_idx):
        x = batch[0]
        recon_x, mu, L = self.forward(x)
        loss, recon_loss, kl = self._vae_loss(x, recon_x, mu, L)
        self.log_dict({"train_loss": loss, "recon_loss": recon_loss, "kl_div": kl}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        recon_x, mu, L = self(x)
        loss, recon_loss, kl = self._vae_loss(x, recon_x, mu, L)
        self.log_dict({"val_loss": loss, "val_recon_loss": recon_loss, "val_kl_div": kl})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
