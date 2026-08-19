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


def test_decode_returns_unbounded_logits_not_probabilities() -> None:
    model = SimpleVAE(input_dim=6, hidden_dims=[5], latent_dim=2)
    with torch.no_grad():
        for layer in model.decoder:
            if isinstance(layer, torch.nn.Linear):
                layer.bias.fill_(50.0)

    logits = model.decode(torch.zeros(3, 2))

    assert not isinstance(model.decoder[-1], torch.nn.Sigmoid)
    assert logits.max() > 1.0, "decode() must return logits, not probabilities"

    probabilities = model.decode_probabilities(torch.zeros(3, 2))
    assert probabilities.shape == logits.shape
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    torch.testing.assert_close(probabilities, torch.sigmoid(logits))


def test_bce_with_logits_stays_finite_where_plain_bce_diverges() -> None:
    """Saturated reconstructions must still yield finite gradients."""

    target = torch.tensor([[1.0, 0.0]])
    saturated_logits = torch.tensor([[-40.0, 40.0]], requires_grad=True)

    loss = torch.nn.BCEWithLogitsLoss(reduction="sum")(saturated_logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.all(torch.isfinite(saturated_logits.grad))

    # The equivalent probability-space computation is what the old
    # Sigmoid + BCELoss pairing performed, and it is not finite.
    probabilities = torch.sigmoid(saturated_logits.detach())
    unstable_gradient = (probabilities - target) / (probabilities * (1 - probabilities))
    assert not torch.all(torch.isfinite(unstable_gradient))


def test_posterior_scale_tril_is_finite_for_extreme_covariance_parameters() -> None:
    """An encoder that saturates must not produce a non-finite posterior scale.

    Run I2 of the BCE-fix sweep went NaN at epoch 27 through this path, with
    BCEWithLogitsLoss already in place, because ``torch.exp`` overflows float32
    around an argument of 88 and nothing bounded the encoder output.
    """
    for covariance_type, parameter_dim in (("isotropic", 1), ("diagonal", 2), ("full", 3)):
        model = SimpleVAE(input_dim=6, hidden_dims=[5], latent_dim=2, covariance_type=covariance_type)
        extreme = torch.tensor([[500.0] * parameter_dim, [-500.0] * parameter_dim])

        scale_tril = model.posterior_scale_tril(extreme)

        assert torch.isfinite(scale_tril).all(), f"{covariance_type} produced a non-finite scale"


def test_reparameterize_stays_finite_under_saturated_encoder_output() -> None:
    """The clamp has to hold through sampling, not only in the scale itself."""
    model = SimpleVAE(input_dim=6, hidden_dims=[5], latent_dim=2, covariance_type="full")
    mu = torch.zeros(2, 2)
    extreme = torch.full((2, 3), 500.0)

    z = model.reparameterize(mu, extreme)

    assert torch.isfinite(z).all()
