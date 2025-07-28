import torch
from torch import nn


class SimpleVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        """
        A simple Variational Autoencoder (VAE) implementation.
        Args:
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Dimension of the hidden layer.
            latent_dim (int): Dimension of the latent space.
        """
        super(SimpleVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # Output mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # Assuming input is normalized between 0 and 1
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var


class ConvolutionalVAE(nn.Module):
    def __init__(self, input_channels: int, latent_dim: int, image_size: int = 28):
        super(ConvolutionalVAE, self).__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Calculate intermediate dimensions
        # After two stride=2 convolutions: size // 4
        self.encoded_size = image_size // 4
        self.encoded_dim = 64 * self.encoded_size * self.encoded_size

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.encoded_dim, latent_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.encoded_dim),
            nn.ReLU(),
            nn.Unflatten(1, (64, self.encoded_size, self.encoded_size)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # Assuming input is normalized between 0 and 1
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        print("mu shape:", mu.shape, "log_var shape:", log_var.shape)
        z = self.reparameterize(mu, log_var)
        print("z shape:", z.shape)
        recon_x = self.decode(z)
        return recon_x, mu, log_var


# provide example usage of the ConvolutionalVAE
if __name__ == "__main__":
    # Example usage
    input_channels = 1  # For grayscale images like MNIST
    latent_dim = 2
    model = ConvolutionalVAE(input_channels, latent_dim)

    # Create a random input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(16, input_channels, 28, 28)  # Batch size of 16

    # Forward pass
    recon_x, mu, log_var = model(input_tensor)
    print("Reconstructed shape:", recon_x.shape)
    print("Mean shape:", mu.shape)
    print("Log variance shape:", log_var.shape)
