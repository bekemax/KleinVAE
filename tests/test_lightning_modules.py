import pytest
import torch

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

    recon_x, mu, log_var = module(x)

    assert recon_x.shape == x.shape
    assert mu.shape == (4, 2)
    assert log_var.shape == (4, 1)
    assert recon_x.device.type == "cpu"


@pytest.mark.parametrize("module_cls", [VAEModule, TorusVAEModule, KleinVAEModule])
def test_lightning_module_vae_loss_returns_scalar_tensors(module_cls: type[VAEModule]) -> None:
    module = _module(module_cls)
    x = torch.rand(4, 6)
    recon_x, mu, log_var = module(x)

    loss, recon_loss, kl_loss = module._vae_loss(x, recon_x, mu, log_var)

    assert loss.ndim == 0
    assert recon_loss.ndim == 0
    assert kl_loss.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(recon_loss)
    assert torch.isfinite(kl_loss)


def test_vanilla_kl_is_zero_for_standard_normal_posterior() -> None:
    module = _module(VAEModule)
    mu = torch.zeros(4, 2)
    log_var = torch.zeros(4, 1)

    kl_loss = module.kl_loss(mu, log_var)

    torch.testing.assert_close(kl_loss, torch.tensor(0.0), atol=1e-7, rtol=0)

