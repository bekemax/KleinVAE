import pytest
import torch
from torch.distributions import MultivariateNormal, kl_divergence
from types import SimpleNamespace

from src.callbacks.topology_metrics import TopologyMetricsCallback
from src.models.components.vae import SimpleVAE
from src.models.klein_vae_module import KleinVAEModule
from src.models.torus_vae_module import TorusVAEModule
from src.models.vae_module import VAEModule


def _base_model() -> SimpleVAE:
    return SimpleVAE(input_dim=6, hidden_dims=[5], latent_dim=2)


def _module(module_cls: type[VAEModule]) -> VAEModule:
    return module_cls(
        model=_base_model(),
        optimizer=torch.optim.Adam,
        recon_loss=torch.nn.MSELoss(reduction="mean"),
    )


@pytest.mark.parametrize("module_cls", [VAEModule, TorusVAEModule, KleinVAEModule])
def test_lightning_module_forward_shapes_on_cpu(module_cls: type[VAEModule]) -> None:
    module = _module(module_cls)
    x = torch.rand(4, 6)

    recon_x, mu, covariance_parameters = module(x)

    assert recon_x.shape == x.shape
    assert mu.shape == (4, 2)
    assert covariance_parameters.shape == (4, 3)
    assert recon_x.device.type == "cpu"


@pytest.mark.parametrize("module_cls", [VAEModule, TorusVAEModule, KleinVAEModule])
def test_lightning_module_vae_loss_returns_scalar_tensors(module_cls: type[VAEModule]) -> None:
    module = _module(module_cls)
    x = torch.rand(4, 6)
    recon_x, mu, covariance_parameters = module(x)

    loss, recon_loss, kl_loss = module._vae_loss(
        x, recon_x, mu, covariance_parameters
    )

    assert loss.ndim == 0
    assert recon_loss.ndim == 0
    assert kl_loss.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(recon_loss)
    assert torch.isfinite(kl_loss)


def test_vanilla_kl_is_zero_for_standard_normal_posterior() -> None:
    module = _module(VAEModule)
    mu = torch.zeros(4, 2)
    covariance_parameters = torch.zeros(4, 3)

    kl_loss = module.kl_loss(mu, covariance_parameters)

    torch.testing.assert_close(kl_loss, torch.tensor(0.0), atol=1e-7, rtol=0)


def test_vanilla_kl_supports_configured_narrow_prior() -> None:
    module = VAEModule(
        model=_base_model(),
        optimizer=torch.optim.Adam,
        recon_loss=torch.nn.MSELoss(reduction="sum"),
        prior_mean=0.0,
        prior_variance=0.01,
    )
    mu = torch.zeros(4, 2)
    log_scale = torch.log(torch.tensor(0.1))
    covariance_parameters = torch.tensor([log_scale, 0.0, log_scale]).repeat(4, 1)

    torch.testing.assert_close(
        module.kl_loss(mu, covariance_parameters),
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0,
    )


def test_topological_kl_is_zero_when_posterior_matches_manuscript_prior() -> None:
    module = _module(KleinVAEModule)
    mu = torch.full((4, 2), 0.5)
    log_scale = torch.log(torch.tensor(0.1).sqrt())
    covariance_parameters = torch.tensor([log_scale, 0.0, log_scale]).repeat(4, 1)

    kl_loss = module.kl_loss(mu, covariance_parameters)

    torch.testing.assert_close(kl_loss, torch.tensor(0.0), atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    ("module_cls", "prior_mean", "prior_variance"),
    [(VAEModule, 0.0, 1.0), (KleinVAEModule, 0.5, 0.1)],
)
def test_analytic_full_covariance_kl_matches_torch_distribution(
    module_cls: type[VAEModule], prior_mean: float, prior_variance: float
) -> None:
    module = _module(module_cls)
    mu = torch.tensor([[0.2, -0.4], [0.7, 0.1]])
    covariance_parameters = torch.tensor(
        [
            [torch.log(torch.tensor(0.7)), 0.2, torch.log(torch.tensor(1.1))],
            [torch.log(torch.tensor(0.4)), -0.3, torch.log(torch.tensor(0.8))],
        ]
    )
    scale_tril = module.model.posterior_scale_tril(covariance_parameters)
    expected = kl_divergence(
        MultivariateNormal(mu, scale_tril=scale_tril),
        MultivariateNormal(
            torch.full_like(mu, prior_mean),
            covariance_matrix=torch.eye(2).expand(2, 2, 2) * prior_variance,
        ),
    ).mean()

    actual = module.kl_loss(mu, covariance_parameters)

    torch.testing.assert_close(actual, expected)


def test_negative_elbo_is_unweighted_even_for_beta_objective() -> None:
    module = _module(VAEModule)
    module.kl_weight = 0.25
    x = torch.rand(4, 6)
    reconstruction, mu, covariance_parameters = module(x)

    objective, reconstruction_loss, kl_loss = module._vae_loss(
        x, reconstruction, mu, covariance_parameters
    )

    torch.testing.assert_close(objective, reconstruction_loss + 0.25 * kl_loss)
    assert not torch.allclose(objective, reconstruction_loss + kl_loss)


def test_mean_reconstruction_loss_is_rescaled_for_negative_elbo_logging() -> None:
    module = VAEModule(
        model=_base_model(),
        optimizer=torch.optim.Adam,
        recon_loss=torch.nn.BCELoss(reduction="mean"),
    )
    x = torch.rand(4, 6)
    reconstruction = torch.rand(4, 6)

    configured_loss = module.reconstruction_loss(x, reconstruction)
    reconstruction_nll = module.reconstruction_nll(x, reconstruction)

    torch.testing.assert_close(reconstruction_nll, configured_loss * 6)


@pytest.mark.parametrize("module_cls", [VAEModule, TorusVAEModule, KleinVAEModule])
def test_empirical_latent_code_variances_are_finite(
    module_cls: type[VAEModule],
) -> None:
    module = _module(module_cls)
    trainer = SimpleNamespace(
        datamodule=SimpleNamespace(data_for_pd=torch.rand(32, 6))
    )
    callback = TopologyMetricsCallback(save_figures=False, save_artifacts=False)

    variance = callback._empirical_latent_variances(trainer, module)

    assert variance.ndim == 0
    assert torch.isfinite(variance) and variance >= 0
