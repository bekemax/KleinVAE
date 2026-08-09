import numpy as np
import pytest
import torch

from src.utils.topology_utils import (
    compute_pairwise_bottlenecks,
    compute_normalized_persistence_diagrams,
    compute_persistence_diagrams,
    compute_topology_bottlenecks,
    euclidean_distance_matrix,
    klein_distance_matrix,
    normalize_distance_matrix,
    preimage_of_klein,
    preimage_of_torus_grid,
    project_to_klein,
    project_to_torus,
    torus_distance_matrix,
)


def _assert_contains_point(points: torch.Tensor, expected: torch.Tensor, *, atol: float = 1e-6) -> None:
    distances = torch.linalg.norm(points - expected, dim=-1)
    assert torch.any(distances <= atol), f"{expected.tolist()} was not found in {points.tolist()}"


def test_project_to_torus_maps_points_to_fundamental_domain() -> None:
    points = torch.tensor(
        [
            [-0.25, -0.1],
            [2.0, 1.0],
            [5.5, 3.25],
            [1.5, 0.25],
        ]
    )

    projected = project_to_torus(points, stack=True)

    assert torch.all(projected[:, 0] >= 0)
    assert torch.all(projected[:, 0] < 2)
    assert torch.all(projected[:, 1] >= 0)
    assert torch.all(projected[:, 1] < 1)
    torch.testing.assert_close(
        projected,
        torch.tensor([[1.75, 0.9], [0.0, 0.0], [1.5, 0.25], [1.5, 0.25]]),
    )


def test_project_to_torus_is_invariant_under_deck_transformations() -> None:
    points = torch.tensor([[0.25, 0.2], [1.75, 0.8], [0.0, 0.0]])
    deck_transformed = points + torch.tensor([[4.0, -3.0], [-2.0, 2.0], [6.0, -4.0]])

    torch.testing.assert_close(project_to_torus(points, stack=True), project_to_torus(deck_transformed, stack=True))


def test_project_to_torus_preserves_points_in_fundamental_domain() -> None:
    points = torch.tensor([[0.0, 0.0], [0.25, 0.5], [1.999, 0.999]])

    torch.testing.assert_close(project_to_torus(points, stack=True), points)


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ([0.25, 0.5], [0.25, 0.5]),
        ([1.25, 0.2], [0.25, 0.8]),
        ([-0.25, 1.2], [0.75, 0.8]),
        ([3.25, -0.3], [0.25, 0.3]),
        ([1.25, 0.0], [0.25, 0.0]),
    ],
)
def test_project_to_klein_maps_explicit_points_to_fundamental_domain(point: list[float], expected: list[float]) -> None:
    projected = project_to_klein(torch.tensor([point]))[0]

    assert 0 <= projected[0] < 1
    assert 0 <= projected[1] < 1
    torch.testing.assert_close(projected, torch.tensor(expected), atol=1e-6, rtol=0)


def test_project_to_klein_is_invariant_under_klein_identifications() -> None:
    points = torch.tensor([[0.2, 0.3], [0.7, 0.9], [0.4, 0.0]])
    vertical_period = points + torch.tensor([0.0, 2.0])
    twisted_period = torch.stack([points[:, 0] + 1.0, 1.0 - points[:, 1]], dim=-1)

    projected = project_to_klein(points)
    torch.testing.assert_close(projected, project_to_klein(vertical_period), atol=1e-6, rtol=0)
    torch.testing.assert_close(projected, project_to_klein(twisted_period), atol=1e-6, rtol=0)


def test_preimage_of_klein_returns_two_expected_equivalent_representatives() -> None:
    points = torch.tensor([[0.2, 0.3], [0.75, 0.1], [0.4, 0.0]])

    preimages = preimage_of_klein(points)

    assert preimages.shape == (3, 2, 2)
    torch.testing.assert_close(preimages[:, 0, :], points)
    torch.testing.assert_close(preimages[:, 1, 0], points[:, 0] + 1.0)
    torch.testing.assert_close(preimages[:, 1, 1], 1.0 - points[:, 1])

    for point, lifts in zip(points, preimages):
        projected_lifts = project_to_klein(lifts)
        torch.testing.assert_close(projected_lifts, point.expand_as(projected_lifts), atol=1e-6, rtol=0)


def test_preimage_of_torus_grid_contains_original_and_periodic_lifts() -> None:
    points = torch.tensor([[0.25, 0.4], [1.5, 0.75]])

    lifts = preimage_of_torus_grid(points, grid_size=1)

    assert lifts.shape == (2, 9, 2)
    _assert_contains_point(lifts[0], torch.tensor([0.25, 0.4]))
    _assert_contains_point(lifts[0], torch.tensor([2.25, 0.4]))
    _assert_contains_point(lifts[0], torch.tensor([-1.75, 0.4]))
    _assert_contains_point(lifts[0], torch.tensor([0.25, 1.4]))
    _assert_contains_point(lifts[0], torch.tensor([0.25, -0.6]))


def test_torus_distance_matrix_has_metric_symmetries_and_known_wraparound_distances() -> None:
    points = torch.tensor(
        [
            [0.05, 0.5],
            [1.95, 0.5],
            [1.0, 0.02],
            [1.0, 0.98],
        ]
    )

    distances = torus_distance_matrix(points)

    torch.testing.assert_close(distances, distances.T)
    torch.testing.assert_close(torch.diag(distances), torch.zeros(points.shape[0]))
    torch.testing.assert_close(distances[0, 1], torch.tensor(0.1), atol=1e-6, rtol=0)
    torch.testing.assert_close(distances[2, 3], torch.tensor(0.04), atol=1e-6, rtol=0)


def test_torus_distance_matrix_is_invariant_under_equivalent_representatives() -> None:
    points = torch.tensor([[0.2, 0.3], [1.8, 0.9], [1.0, 0.1]])
    equivalent = points + torch.tensor([[2.0, 1.0], [-2.0, 0.0], [0.0, -1.0]])

    torch.testing.assert_close(torus_distance_matrix(points), torus_distance_matrix(equivalent), atol=1e-6, rtol=0)


def test_klein_distance_matrix_uses_twisted_identification() -> None:
    points = torch.tensor(
        [
            [0.05, 0.2],
            [0.95, 0.8],
            [0.4, 0.4],
        ]
    )

    distances = klein_distance_matrix(points)

    torch.testing.assert_close(distances, distances.T)
    torch.testing.assert_close(torch.diag(distances), torch.zeros(points.shape[0]))
    torch.testing.assert_close(distances[0, 1], torch.tensor(0.1), atol=1e-6, rtol=0)
    assert distances[0, 1] < torch.linalg.norm(points[0] - points[1])


def test_klein_distance_matrix_is_invariant_under_equivalent_representatives() -> None:
    points = torch.tensor([[0.05, 0.2], [0.95, 0.8], [0.4, 0.4]])
    equivalent = points.clone()
    equivalent[1] = torch.tensor([1.95, 0.2])

    torch.testing.assert_close(klein_distance_matrix(points), klein_distance_matrix(equivalent), atol=1e-6, rtol=0)


def test_compute_persistence_diagrams_returns_expected_structure() -> None:
    point_cloud = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    diagrams = compute_persistence_diagrams(point_cloud, maxdim=1, coeffs=[2, 3])

    assert set(diagrams) == {2, 3}
    for diagram_list in diagrams.values():
        assert isinstance(diagram_list, list)
        assert len(diagram_list) == 2
        assert all(diagram.ndim == 2 and diagram.shape[1] == 2 for diagram in diagram_list)
        assert diagram_list[0].shape[0] == point_cloud.shape[0]


def test_compute_pairwise_bottlenecks_for_identical_and_perturbed_diagrams() -> None:
    original = {
        2: [
            np.array([[0.0, 0.5], [0.0, 0.75]]),
            np.array([[0.2, 0.6]]),
        ]
    }
    identical = {2: [diagram.copy() for diagram in original[2]]}
    perturbed = {
        2: [
            np.array([[0.0, 0.55], [0.0, 0.7]]),
            np.array([[0.25, 0.65]]),
        ]
    }

    identical_distances = compute_pairwise_bottlenecks(original, identical)
    perturbed_distances = compute_pairwise_bottlenecks(original, perturbed)

    np.testing.assert_allclose(identical_distances[2], np.zeros(2))
    assert np.all(np.isfinite(perturbed_distances[2]))
    assert np.all(perturbed_distances[2] >= 0)


def test_normalize_distance_matrix_removes_global_scale() -> None:
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    distances = euclidean_distance_matrix(points)

    normalized = normalize_distance_matrix(distances, quantile=0.75)
    rescaled = normalize_distance_matrix(17.0 * distances, quantile=0.75)

    torch.testing.assert_close(normalized, rescaled)
    torch.testing.assert_close(normalized, normalized.T)
    torch.testing.assert_close(torch.diag(normalized), torch.zeros(points.shape[0]))


def test_normalize_distance_matrix_rejects_degenerate_metric() -> None:
    with pytest.raises(ValueError, match="positive distance"):
        normalize_distance_matrix(torch.zeros(3, 3))


def test_normalized_persistence_and_topology_score_are_scale_invariant() -> None:
    points = torch.tensor(
        [[np.cos(angle), np.sin(angle)] for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False)],
        dtype=torch.float32,
    )
    distances = euclidean_distance_matrix(points)
    original = compute_normalized_persistence_diagrams(distances, maxdim=1, coeffs=[2])
    rescaled = compute_normalized_persistence_diagrams(5.0 * distances, maxdim=1, coeffs=[2])

    bottlenecks, score = compute_topology_bottlenecks(
        original,
        rescaled,
        homology_dimensions=(1,),
    )

    assert bottlenecks[2][1] == pytest.approx(0.0, abs=1e-7)
    assert score == pytest.approx(0.0, abs=1e-7)
