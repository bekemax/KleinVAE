import pytest
import torch
from torch.distributions import MultivariateNormal, kl_divergence
from typing import Any
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
        recon_loss=torch.nn.BCEWithLogitsLoss(reduction="mean"),
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


def test_topology_metrics_callback_runs_on_test_epoch_end() -> None:
    module = _module(KleinVAEModule)
    data = torch.rand(16, 6)
    params = torch.rand(16, 2)
    datamodule = SimpleNamespace(
        data_for_pd=data,
        validation_data_for_pd=data,
        validation_parameters_for_pd=params,
        test_data_for_pd=data,
        test_parameters_for_pd=params,
    )
    logged: dict[str, Any] = {}
    module.log = lambda key, val, **kwargs: logged.update({key: val})
    module.print = lambda *args, **kwargs: None
    trainer = SimpleNamespace(
        datamodule=datamodule,
        current_epoch=0,
        max_epochs=1,
        global_rank=0,
        default_root_dir="/tmp/test_topology_callback",
        loggers=[],
    )
    callback = TopologyMetricsCallback(
        save_figures=False,
        save_artifacts=False,
        topology_dimensions=[1],
        field_orders=[2],
    )
    callback.on_test_epoch_end(trainer, module)
    assert "test_latent_code_variance" in logged
    assert "test_latent_topology_score" in logged
    assert "test_bottleneck_over_2_dim_0" in logged
    assert "test_total_bottleneck_over_2" in logged
    assert callback.test_latent_topology_score is not None


def _logging_module(module_cls: type[VAEModule] = VAEModule) -> tuple[VAEModule, dict[str, Any]]:
    module = _module(module_cls)
    logged: dict[str, Any] = {}
    module.log_dict = lambda metrics, **kwargs: logged.update(metrics)
    return module, logged


@pytest.mark.parametrize("stage", ["val", "test"])
def test_shared_step_separates_posterior_mean_nll_from_the_elbo_term(stage: str) -> None:
    """recon_nll must mean what evaluate_checkpoint.py means by it: the deterministic one."""
    torch.manual_seed(0)
    module, logged = _logging_module()
    x = torch.rand(4, 6)

    module._shared_step(stage, (x,))

    assert f"{stage}_recon_nll" in logged and f"{stage}_elbo_recon_nll" in logged
    mu, _ = module.model.encode(x)
    with torch.no_grad():
        expected = module.reconstruction_nll(x, module.model.decode(module.project_latent(mu)))
    torch.testing.assert_close(logged[f"{stage}_recon_nll"], expected)
    # The sampled term feeds the ELBO; the deterministic one must not.
    torch.testing.assert_close(logged[f"{stage}_negative_elbo"], logged[f"{stage}_elbo_recon_nll"] + logged[f"{stage}_kl_loss"])


def test_training_step_skips_the_extra_deterministic_decode() -> None:
    """Training throughput matters more than a metric nothing reads there."""
    module, logged = _logging_module()

    module._shared_step("train", (torch.rand(4, 6),))

    assert "train_elbo_recon_nll" in logged
    assert "train_recon_nll" not in logged


def _pd_datamodule(**overrides: Any) -> SimpleNamespace:
    data, params = torch.rand(12, 6), torch.rand(12, 2)
    fields = dict(
        data_for_pd=data,
        validation_parameters_for_pd=params,
        test_data_for_pd=data,
        test_parameters_for_pd=params,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_topology_callback_rejects_an_unknown_stage() -> None:
    callback = TopologyMetricsCallback()
    trainer = SimpleNamespace(datamodule=_pd_datamodule())

    with pytest.raises(ValueError, match="Unknown stage"):
        callback._get_normalized_original_diagrams(trainer, stage="validation")


def test_topology_callback_fails_loudly_when_test_parameters_are_missing() -> None:
    """Falling back to the validation parameters would pair them with the test point cloud."""
    callback = TopologyMetricsCallback(input_metric="intrinsic_klein", field_orders=[2])
    trainer = SimpleNamespace(datamodule=_pd_datamodule(test_parameters_for_pd=None))

    with pytest.raises(ValueError, match="test_parameters_for_pd"):
        callback._get_normalized_original_diagrams(trainer, stage="test")


def test_topology_callback_uses_validation_parameters_for_the_val_stage() -> None:
    callback = TopologyMetricsCallback(input_metric="intrinsic_klein", field_orders=[2])
    trainer = SimpleNamespace(datamodule=_pd_datamodule(validation_parameters_for_pd=None))

    with pytest.raises(ValueError, match="validation_parameters_for_pd"):
        callback._get_normalized_original_diagrams(trainer, stage="val")
