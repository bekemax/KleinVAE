from typing import Tuple

import torch

from src.models.components.vae import SimpleVAE
from src.models.torus_vae_module import TorusVAEModule
from src.utils.topology_utils import project_to_klein


class KleinVAEModule(TorusVAEModule):
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.model.encode(x)
        z_on_plane = self.model.reparameterize(mu, log_var)
        z_on_klein = project_to_klein(z_on_plane)
        recon_x = self.model.decode(z_on_klein)
        return recon_x, mu, log_var


if __name__ == "__main__":
    model = SimpleVAE(input_dim=28 * 28, hidden_dims=[128, 64])
    vae_module = KleinVAEModule(model=model, optimizer=torch.optim.Adam)

    x = torch.randn(16, 28 * 28)
    recon_x, mu, log_var = vae_module(x)
    print(f"Reconstructed x shape: {recon_x.shape}, mu shape: {mu.shape}, log_var shape: {log_var.shape}")
