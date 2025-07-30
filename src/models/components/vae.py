from torch import nn
import torch

from typing import List


class SimpleVAE(nn.Module):
    def __init__(self, input_dim: int = 9, hidden_dims: List[int] = [16, 32, 64]):
        """
        A Variational Autoencoder (VAE) implementation.
        Args:
            input_dim (int): Dimension of the input data.
            hidden_dims (List[int]): List of hidden layer dimensions (same for encoder and decoder, but reversed for decoder).
        Encoder output is fixed to 5 (for 2D mean, and LT matrix L s.t. \Sigma = LL^T).
        Decoder input is fixed to 2 (z is always 2D).
        """
        super(SimpleVAE, self).__init__()
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.LeakyReLU())
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, 5))  # Output: 2 mean, sigma_X, sigma_Y, non_diag
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = 4  # Decoder input is always 2 (z)
        for h_dim in [4, 8, 16]:  # reversed(hidden_dims)
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.LeakyReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())  # Assuming input is normalized between 0 and 1
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        # Output: [mu_x, mu_y, log_sigma_x, log_sigma_y, non_diag]
        mu = h[..., :2]
        log_sigma = h[..., 2:4]
        non_diag = h[..., 4:5]
        return mu, log_sigma, non_diag

    def reparameterize(self, mu, log_sigma, rho):
        raise NotImplementedError("Reparameterization is implemented in the LightningModule.")

    def decode(self, z):
        y = z * 2 * torch.pi  # Scale z to [0, 2pi)
        sin_cos = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)
        return self.decoder(sin_cos)

    def forward(self, x):
        mu, log_sigma, rho = self.encode(x)
        z = self.reparameterize(mu, log_sigma, rho)
        recon_x = self.decode(z)
        return recon_x, mu, log_sigma, rho


class ConvolutionalVAE(nn.Module):
    def __init__(self, input_channels: int = 1, hidden_dims: List[int] = [32, 64, 128], latent_dim: int = 2, image_size: int = 28):
        """
        A flexible Convolutional Variational Autoencoder (VAE) implementation.
        Args:
            input_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
            hidden_dims (List[int]): List of hidden layer dimensions for encoder/decoder.
            latent_dim (int): Dimension of the latent space (default 2 for 2D).
            image_size (int): Size of the input image (assumed square).
        Encoder output is fixed to 5 (for 2D mean, and LT matrix L s.t. \Sigma = LL^T).
        Decoder input is fixed to 2 (z is always 2D).
        """
        super(ConvolutionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.input_channels = input_channels

        # Encoder
        encoder_layers = []
        in_channels = input_channels
        for h_dim in hidden_dims:
            encoder_layers.extend([nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.ReLU()])
            in_channels = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Calculate the size of the feature map after encoding
        self.feature_size = image_size // (2 ** len(hidden_dims))
        self.feature_dim = hidden_dims[-1] * self.feature_size * self.feature_size

        # Latent space projection
        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_log_sigma = nn.Linear(self.feature_dim, latent_dim)
        self.fc_non_diag = nn.Linear(self.feature_dim, 1)

        # Decoder
        decoder_layers = []
        in_channels = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                ]
            )
            in_channels = h_dim

        # Final layer to get back to input channels
        decoder_layers.extend(
            [
                nn.ConvTranspose2d(in_channels, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid(),  # Assuming input is normalized between 0 and 1
            ]
        )

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)  # Flatten for FC layers

        mu = self.fc_mu(h_flat)
        log_sigma = self.fc_log_sigma(h_flat)
        non_diag = self.fc_non_diag(h_flat)

        return mu, log_sigma, non_diag

    def reparameterize(self, mu, log_sigma, rho):
        raise NotImplementedError("Reparameterization is implemented in the LightningModule.")

    def decode(self, z):
        # Reshape z to start the deconvolution process
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.decoder(z)

    def forward(self, x):
        mu, log_sigma, rho = self.encode(x)
        z = self.reparameterize(mu, log_sigma, rho)
        recon_x = self.decode(z)
        return recon_x, mu, log_sigma, rho
