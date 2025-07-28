from src.utils import project_to_klein

import lightning as pl
import torch
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal


class KleinVAEModule(pl.LightningModule):
    def __init__(self, model, std=1.0, lr=1e-3, batch_size: int = 1):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.batch_size = batch_size

    def sample(self, num_samples):
        return project_to_klein(torch.randn(num_samples, 2))

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.model.encode(x)
        z_on_plane = self.reparameterize(mu, log_var)
        z_on_klein = project_to_klein(z_on_plane)
        recon_x = self.model.decode(z_on_klein)
        return recon_x, mu, log_var

    def _vae_loss(self, x, recon_x, mu, log_var):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
        var = torch.exp(0.5 * log_var)
        q = Normal(mu, var)
        # Create prior with the correct batch size and device
        prior_loc = torch.zeros_like(mu)
        prior_scale = torch.ones_like(var)
        prior_dist = Normal(prior_loc, prior_scale)
        kl_div = kl_divergence(q, prior_dist).mean()
        return recon_loss + kl_div, recon_loss, kl_div

    def training_step(self, batch, batch_idx):
        x = batch[0]
        # x = x.view(x.size(0), -1)  # Flatten MNIST image
        recon_x, mu, log_var = self(x)
        loss, recon_loss, kl = self._vae_loss(x, recon_x, mu, log_var)
        self.log_dict({"train_loss": loss, "recon_loss": recon_loss, "kl_div": kl})
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        recon_x, mu, log_var = self(x)
        loss, recon_loss, kl = self._vae_loss(x, recon_x, mu, log_var)
        self.log_dict({"val_loss": loss, "val_recon_loss": recon_loss, "val_kl_div": kl})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
