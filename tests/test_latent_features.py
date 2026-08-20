import math

import pytest
import torch

from src.models.components.latent_features import (
    KleinFourierFeatures,
    RandomFourierFeatures,
    TorusFourierFeatures,
)
from src.models.components.vae import SimpleVAE
from src.utils.topology_utils import KLEIN_U_WIDTH, TORUS_U_PERIOD, TORUS_V_PERIOD, project_to_klein


def _latents(n: int = 256, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 2, generator=generator) - 0.5) * 6


def test_klein_features_are_invariant_under_both_deck_generators() -> None:
    """The whole point of the lift: the two identifications must be exact symmetries."""
    latents = _latents()
    features = KleinFourierFeatures()(latents)

    v_shift = latents + torch.tensor([0.0, TORUS_V_PERIOD])
    twist = torch.stack([latents[:, 0] + KLEIN_U_WIDTH, -latents[:, 1]], dim=-1)

    torch.testing.assert_close(KleinFourierFeatures()(v_shift), features)
    torch.testing.assert_close(KleinFourierFeatures()(twist), features)


def test_klein_features_separate_distinct_points_of_the_quotient() -> None:
    """Invariance is worthless without injectivity, or the decoder could not tell points apart."""
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0, KLEIN_U_WIDTH, 17)[:-1],
            torch.linspace(0, TORUS_V_PERIOD, 17)[:-1],
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    features = KleinFourierFeatures()(grid)

    distances = torch.cdist(features, features)
    distances.fill_diagonal_(float("inf"))

    assert distances.min() > 1e-3, "distinct fundamental-domain points collapsed to the same feature"


def test_klein_features_ignore_the_fundamental_domain_seam() -> None:
    """Points straddling a seam are neighbours in feature space though not in domain coordinates."""
    inside = torch.tensor([[KLEIN_U_WIDTH - 1e-5, 0.3]])
    outside = torch.tensor([[KLEIN_U_WIDTH + 1e-5, 0.3]])

    feature_gap = (KleinFourierFeatures()(inside) - KleinFourierFeatures()(outside)).norm()
    coordinate_gap = (project_to_klein(inside) - project_to_klein(outside)).norm()

    assert feature_gap < 1e-3
    assert coordinate_gap > 0.3, "the seam should be a large jump in raw coordinates"


def test_klein_features_are_unchanged_by_projecting_into_the_fundamental_domain() -> None:
    """Invariance holds on all of R^2, so projecting first is a no-op for the decoder."""
    latents = _latents()

    torch.testing.assert_close(
        KleinFourierFeatures()(latents), KleinFourierFeatures()(project_to_klein(latents))
    )


def test_torus_features_are_invariant_under_both_periods() -> None:
    latents = _latents()
    features = TorusFourierFeatures()(latents)

    torch.testing.assert_close(TorusFourierFeatures()(latents + torch.tensor([TORUS_U_PERIOD, 0.0])), features)
    torch.testing.assert_close(TorusFourierFeatures()(latents + torch.tensor([0.0, TORUS_V_PERIOD])), features)


def test_torus_features_do_not_identify_the_klein_twist() -> None:
    """Torus and Klein differ only in this identification, which is what the ablation contrasts."""
    latents = _latents(n=32)
    twist = torch.stack([latents[:, 0] + KLEIN_U_WIDTH, -latents[:, 1]], dim=-1)

    assert not torch.allclose(TorusFourierFeatures()(twist), TorusFourierFeatures()(latents), atol=1e-3)


def test_random_features_are_seam_free_but_carry_no_identification() -> None:
    """The Euclidean control: same smooth basis, no quotient."""
    feature_map = RandomFourierFeatures(latent_dim=2, output_dim=5)
    latents = _latents(n=32)
    twist = torch.stack([latents[:, 0] + KLEIN_U_WIDTH, -latents[:, 1]], dim=-1)

    assert feature_map.output_dim == 5
    assert not torch.allclose(feature_map(twist), feature_map(latents), atol=1e-3)


def test_random_features_are_fixed_rather_than_learned() -> None:
    feature_map = RandomFourierFeatures(latent_dim=2, output_dim=5, seed=3)

    assert list(feature_map.parameters()) == []
    torch.testing.assert_close(feature_map.frequencies, RandomFourierFeatures(latent_dim=2, output_dim=5, seed=3).frequencies)


@pytest.mark.parametrize(
    "feature_map,expected_width",
    [(None, 2), (KleinFourierFeatures(), 5), (TorusFourierFeatures(), 4), (RandomFourierFeatures(), 5)],
)
def test_decoder_input_width_follows_the_feature_map(feature_map, expected_width: int) -> None:
    model = SimpleVAE(input_dim=6, hidden_dims=[5], latent_dim=2, latent_feature_map=feature_map)

    assert model.decoder_input_dim == expected_width
    assert model.decoder[0].in_features == expected_width
    assert model.decode(torch.zeros(3, 2)).shape == (3, 6)


def test_decoding_is_continuous_across_the_klein_seam() -> None:
    """A decoder on raw domain coordinates has to jump here; the lifted one must not."""
    model = SimpleVAE(input_dim=6, hidden_dims=[8], latent_dim=2, latent_feature_map=KleinFourierFeatures())
    inside = torch.tensor([[KLEIN_U_WIDTH - 1e-5, 0.3]])
    outside = torch.tensor([[KLEIN_U_WIDTH + 1e-5, 0.3]])

    gap = (model.decode(inside) - model.decode(outside)).abs().max()

    assert gap < 1e-3


def test_klein_features_match_the_closed_form_modes() -> None:
    """Guards the frequency convention: u has period 2, v period 1, twist is a half-shift in u."""
    latent = torch.tensor([[0.37, 0.21]])
    theta, phi = math.pi * 0.37, 2 * math.pi * 0.21

    expected = torch.tensor(
        [[
            math.cos(2 * theta),
            math.sin(2 * theta),
            math.cos(theta) * math.sin(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(phi),
        ]]
    )

    torch.testing.assert_close(KleinFourierFeatures()(latent), expected)
