from torch import nn

from typing import List


class SimpleVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [16, 32, 64], latent_dim: int = 2):
        """
        A Variational Autoencoder (VAE) implementation.
        Args:
            input_dim (int): Dimension of the input data.
            hidden_dims (List[int]): List of hidden layer dimensions (same for encoder and decoder, but reversed for decoder).
        Encoder output is fixed to 3 (for 2D mean, and LT matrix L s.t. \Sigma = LL^T).
        Decoder input is fixed to 2 (z is always 2D).
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
        encoder_layers.append(nn.Linear(prev_dim, latent_dim + 1))  # Output: latent_dim mean, sigma /////_X, sigma_Y, non_diag
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

    def encode(self, x):
        h = self.encoder(x)
        # Output: [mu_x, mu_y, log_sigma_x, log_sigma_y, non_diag]
        mu = h[..., : self.latent_dim]
        log_sigma = h[..., self.latent_dim : self.latent_dim + 1]
        # non_diag = h[..., 4:5]
        return mu, log_sigma, None

    def reparameterize(self, mu, log_sigma, rho):
        raise NotImplementedError("Reparameterization is implemented in the LightningModule.")

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_sigma, rho = self.encode(x)
        z = self.reparameterize(mu, log_sigma, rho)
        recon_x = self.decode(z)
        return recon_x, mu, log_sigma, rho
