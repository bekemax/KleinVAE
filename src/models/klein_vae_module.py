import torch

from src.models.torus_vae_module import TorusVAEModule
from src.utils.topology_utils import klein_distance_matrix, project_to_klein


class KleinVAEModule(TorusVAEModule):
    """VAE using the R^2 -> Klein-bottle covering map."""

    def project_latent(self, z: torch.Tensor) -> torch.Tensor:
        return project_to_klein(z)

    def latent_distance_matrix(self, means: torch.Tensor) -> torch.Tensor:
        return klein_distance_matrix(means)
