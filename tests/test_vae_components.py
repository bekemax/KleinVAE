import torch

from src.models.components.vae import SimpleVAE


def test_simple_vae_forward_returns_expected_shapes() -> None:
    model = SimpleVAE(input_dim=6, hidden_dims=[5], latent_dim=2)
    x = torch.rand(4, 2, 3)

    recon_x, mu, covariance_parameters = model(x)

    assert recon_x.shape == (4, 6)
    assert mu.shape == (4, 2)
    assert covariance_parameters.shape == (4, 3)


def test_simple_vae_reparameterize_preserves_shape_and_device() -> None:
    model = SimpleVAE(input_dim=6, hidden_dims=[5], latent_dim=2)
    mu = torch.zeros(3, 2)
    covariance_parameters = torch.zeros(3, 3)

    z = model.reparameterize(mu, covariance_parameters)

    assert z.shape == mu.shape
    assert z.device == mu.device


def test_full_covariance_parameterization_is_positive_definite() -> None:
    model = SimpleVAE(input_dim=6, hidden_dims=[5], latent_dim=2, covariance_type="full")
    covariance_parameters = torch.tensor([[0.0, 0.5, -0.7]])

    scale_tril = model.posterior_scale_tril(covariance_parameters)
    covariance = scale_tril @ scale_tril.transpose(-1, -2)
    eigenvalues = torch.linalg.eigvalsh(covariance)

    assert scale_tril.shape == (1, 2, 2)
    assert torch.all(torch.diagonal(scale_tril, dim1=-2, dim2=-1) > 0)
    assert torch.all(eigenvalues > 0)
    torch.testing.assert_close(
        model.posterior_log_variance(covariance_parameters),
        torch.log(torch.diagonal(covariance, dim1=-2, dim2=-1)),
    )

def test_simple_vae_decode_matches_flattened_input_dimension() -> None:
    model = SimpleVAE(input_dim=6, hidden_dims=[5], latent_dim=2)
    z = torch.zeros(3, 2)

    decoded = model.decode(z)

    assert decoded.shape == (3, 6)
