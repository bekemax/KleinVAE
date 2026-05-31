import torch

from src.models.components.vae import SimpleVAE


def test_simple_vae_forward_returns_expected_shapes() -> None:
    model = SimpleVAE(input_dim=6, hidden_dims=[5], latent_dim=2)
    x = torch.rand(4, 2, 3)

    recon_x, mu, log_var = model(x)

    assert recon_x.shape == (4, 6)
    assert mu.shape == (4, 2)
    assert log_var.shape == (4, 1)


def test_simple_vae_reparameterize_preserves_shape_and_device() -> None:
    model = SimpleVAE(input_dim=6, hidden_dims=[5], latent_dim=2)
    mu = torch.zeros(3, 2)
    log_var = torch.zeros(3, 1)

    z = model.reparameterize(mu, log_var)

    assert z.shape == mu.shape
    assert z.device == mu.device


def test_simple_vae_decode_matches_flattened_input_dimension() -> None:
    model = SimpleVAE(input_dim=6, hidden_dims=[5], latent_dim=2)
    z = torch.zeros(3, 2)

    decoded = model.decode(z)

    assert decoded.shape == (3, 6)

