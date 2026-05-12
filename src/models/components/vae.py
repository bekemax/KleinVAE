import torch
from torch import nn

from typing import List, Tuple


class SimpleVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [16, 32, 64], latent_dim: int = 2):
        """
        A Variational Autoencoder (VAE) implementation.
        Args:
            input_dim (int): Dimension of the input data.
            hidden_dims (List[int]): List of hidden layer dimensions (same for encoder and decoder, but reversed for decoder).
            latent_dim (int): Dimension of the latent space. Default is 2.

        The encoder outputs the latent mean plus one scalar log-variance. That is an
        isotropic posterior: q(z|x) = N(mu(x), sigma(x)^2 I).
        """
        super(SimpleVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.LeakyReLU())
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim + 1))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim  # Decoder input is always 2 (z)
        for h_dim in reversed(hidden_dims):  # reversed(hidden_dims)
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.LeakyReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())  # Assuming input is normalized between 0 and 1
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.view(x.size(0), -1)
        h = self.encoder(x)
        mu = h[..., : self.latent_dim]
        log_var = h[..., self.latent_dim : self.latent_dim + 1]
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def decode(self, z) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var
