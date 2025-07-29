from src.utils import project_to_klein

import lightning as pl
import torch
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal


class KleinVAEModule(pl.LightningModule):
    def __init__(self, model, lr: float = 1e-3, batch_size: int = 1, kl_weight: float = 1e-1):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.kl_weight = kl_weight

    def sample(self, num_samples):
        return project_to_klein(torch.randn(num_samples, 2))

    def reparameterize(self, mu, log_sigma, non_diag):
        var = torch.exp(log_sigma)
        batch_size = mu.shape[0]

        # Create L matrix for each batch element
        L = torch.zeros(batch_size, 2, 2, device=mu.device)
        L[:, 0, 0] = var[:, 0]  # sigma_x
        L[:, 1, 1] = var[:, 1]  # sigma_y
        L[:, 1, 0] = non_diag[:, 0]  # non-diagonal element

        eps = torch.randn_like(mu)
        return mu + torch.matmul(L, eps.unsqueeze(-1)).squeeze(-1), L

    def forward(self, x):
        mu, log_sigma, non_diag = self.model.encode(x)
        z_on_plane, L = self.reparameterize(mu, log_sigma, non_diag)
        z_on_klein = project_to_klein(z_on_plane)
        recon_x = self.model.decode(z_on_klein)
        return recon_x, mu, L

    def _vae_loss(self, x, recon_x, mu, L):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")

        # Create multivariate normal with batched L matrices
        q = MultivariateNormal(mu, scale_tril=L)

        prior_loc = torch.zeros_like(mu)
        prior_scale = torch.eye(2, device=mu.device).unsqueeze(0).repeat(mu.size(0), 1, 1)
        prior_dist = MultivariateNormal(prior_loc, scale_tril=prior_scale)

        kl_div = kl_divergence(q, prior_dist).mean()
        return recon_loss + self.kl_weight * kl_div, recon_loss, kl_div

    def training_step(self, batch, batch_idx):
        x = batch[0]
        recon_x, mu, L = self(x)
        loss, recon_loss, kl = self._vae_loss(x, recon_x, mu, L)
        self.log_dict({"train_loss": loss, "recon_loss": recon_loss, "kl_div": kl})
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        recon_x, mu, L = self(x)
        loss, recon_loss, kl = self._vae_loss(x, recon_x, mu, L)
        self.log_dict({"val_loss": loss, "val_recon_loss": recon_loss, "val_kl_div": kl})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
