from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from persim import bottleneck, plot_diagrams
from ripser import ripser

TORUS_U_PERIOD = 2.0
TORUS_V_PERIOD = 1.0
KLEIN_U_WIDTH = 1.0


def project_to_torus(points: torch.Tensor, stack: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Given (u, v) in R^2, project each pair (u, v) onto the fundamental domain
    [0, 2) x [0, 1) of the torus by wrapping u and v according to the identifications:
        - (u, v) ~ (u + 2k, v + l) for k,l in Z
    Args:
        points: Tensor of shape (..., 2) representing points in R^2
        stack: If True, return a stacked tensor of shape (..., 2). If False, return a tuple of tensors (u_mod, v_mod).
    Returns:
        If stack is True, a tensor of shape (..., 2) representing points on the torus.
        If stack is False, a tuple of tensors (u_mod, v_mod) each of shape (...,).
    """
    u, v = points[..., 0], points[..., 1]
    u_mod = torch.remainder(u, TORUS_U_PERIOD)
    v_mod = torch.remainder(v, TORUS_V_PERIOD)
    if stack:
        return torch.stack([u_mod, v_mod], dim=-1)
    return u_mod, v_mod


def project_to_klein(points: torch.Tensor) -> torch.Tensor:
    """
    Given (u, v) in R^2, project each pair (u, v) onto the fundamental domain
    [0, 1) x [0, 1) of the Klein bottle by wrapping u and v according to the identifications:
        - (u, v) ~ (u, v + 1)
        - (u, v) ~ (u + 1, 1 - v)
    """

    u_mod, v_mod = project_to_torus(points)

    mask_twist = u_mod >= KLEIN_U_WIDTH
    u_projected = torch.where(mask_twist, u_mod - KLEIN_U_WIDTH, u_mod)
    v_flipped = torch.remainder(TORUS_V_PERIOD - v_mod, TORUS_V_PERIOD)
    v_projected = torch.where(mask_twist, v_flipped, v_mod)

    return torch.stack([u_projected, v_projected], dim=-1)


def preimage_of_klein(points: torch.Tensor) -> torch.Tensor:
    """
    Preimage of points on Klein bottle to points on Torus.
    Args:
        points: Tensor of shape (N, 2) representing points on the Klein bottle
    Returns:
        Tensor of shape (N, 2, 2) representing 2 preimages on the Torus, where
            - the first dimension is the batch size,
            - the second dimension is the preimage index (0 or 1) corresponding to the two possible preimages,
            - the third dimension contains the x and y coordinates.
    """
    xs, ys = points[..., 0], points[..., 1]
    first_preimage = torch.stack([xs, ys], dim=-1)
    second_preimage = torch.stack([xs + KLEIN_U_WIDTH, TORUS_V_PERIOD - ys], dim=-1)
    return torch.stack([first_preimage, second_preimage], dim=-2)


def preimage_of_torus_grid(points, grid_size=1):
    """
    Given points in the torus, return all preimages in a grid around each point.
    Args:
        points: Tensor of shape (..., 2) or (..., k, 2) representing points in the torus.
        grid_size: Size of the grid to generate preimages around each point.
    Returns:
        Tensor of shape (..., k * (2 * grid_size + 1) ** 2, 2) representing nearby preimages in the torus.
    """
    pts = torch.as_tensor(points, dtype=torch.float32)
    if pts.ndim == 2:
        pts = pts.unsqueeze(-2)

    xs, ys = pts[..., 0], pts[..., 1]
    shifts = range(-grid_size, grid_size + 1)
    all_preimages = [torch.stack([xs + TORUS_U_PERIOD * i, ys + TORUS_V_PERIOD * j], dim=-1) for i in shifts for j in shifts]
    return torch.cat(all_preimages, dim=-2)


def preimage_of_torus_recursive(points: torch.Tensor, criterion: Callable[[torch.Tensor], bool]) -> torch.Tensor:
    """
    Given points in the torus, recursively find preimages until the criterion is satisfied.
    The criterion is a function that takes a tensor of points and returns True if the points satisfy the condition.
    """
    grid_x = TORUS_U_PERIOD * torch.linspace(-1, 1, 3, device=points.device)
    grid_y = TORUS_V_PERIOD * torch.linspace(-1, 1, 3, device=points.device)
    meshgrid = torch.meshgrid(grid_x, grid_y, indexing="ij")
    meshgrid_points = torch.column_stack([meshgrid[0].flatten(), meshgrid[1].flatten()])
    preimages = (points[:, None, :] + meshgrid_points[None, :, :]).reshape(-1, 2)
    if criterion(preimages):
        return preimages
    return preimage_of_torus_recursive(preimages, criterion)


def klein_distance_matrix(points: torch.Tensor | np.ndarray, grid_size: int = 1) -> torch.Tensor:
    """
    Compute pairwise distances on the Klein bottle quotient.

    Points are assumed to lie in the Klein fundamental domain [0, 1) x [0, 1).
    Distances are induced from the Euclidean metric on the universal cover by taking
    the minimum over nearby deck-transformed lifts.
    """
    pts = project_to_klein(torch.as_tensor(points, dtype=torch.float32))
    all_lifts = preimage_of_torus_grid(preimage_of_klein(pts), grid_size=grid_size)

    diffs = pts[:, None, None, :] - all_lifts[None, :, :, :]
    distances = torch.linalg.norm(diffs, dim=-1).min(dim=-1).values
    distances = 0.5 * (distances + distances.T)
    diagonal_mask = torch.eye(distances.shape[0], device=distances.device, dtype=torch.bool)
    return distances.masked_fill(diagonal_mask, 0.0)


def torus_distance_matrix(points: torch.Tensor | np.ndarray, grid_size: int = 1) -> torch.Tensor:
    """
    Compute pairwise distances on the torus quotient.

    Points are assumed to lie in the torus fundamental domain [0, 2) x [0, 1).
    Distances are induced from the Euclidean metric on the universal cover by taking
    the minimum over nearby periodic lifts.
    """
    pts = project_to_torus(torch.as_tensor(points, dtype=torch.float32), stack=True)
    all_lifts = preimage_of_torus_grid(pts, grid_size=grid_size)

    diffs = pts[:, None, None, :] - all_lifts[None, :, :, :]
    distances = torch.linalg.norm(diffs, dim=-1).min(dim=-1).values
    distances = 0.5 * (distances + distances.T)
    diagonal_mask = torch.eye(distances.shape[0], device=distances.device, dtype=torch.bool)
    return distances.masked_fill(diagonal_mask, 0.0)


def euclidean_distance_matrix(points: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Compute a symmetric Euclidean pairwise-distance matrix."""

    pts = torch.as_tensor(points, dtype=torch.float32)
    if pts.ndim != 2:
        pts = pts.reshape(pts.shape[0], -1)
    distances = torch.cdist(pts, pts)
    distances = 0.5 * (distances + distances.T)
    diagonal_mask = torch.eye(distances.shape[0], device=distances.device, dtype=torch.bool)
    return distances.masked_fill(diagonal_mask, 0.0)


def normalize_distance_matrix(
    distance_matrix: torch.Tensor | np.ndarray,
    *,
    quantile: float = 0.9,
) -> torch.Tensor:
    """Remove arbitrary metric units using a robust pairwise-distance scale.

    Persistence diagrams from image space and a two-dimensional latent space
    otherwise live in incomparable units.  Dividing each metric by its own
    upper-triangle distance quantile retains relative geometry and topology
    while making cross-space bottleneck distances meaningful.  Zero distances
    are excluded so duplicate observations do not collapse the scale.
    """

    if not 0 < quantile <= 1:
        raise ValueError("quantile must lie in (0, 1]")

    distances = torch.as_tensor(distance_matrix, dtype=torch.float32)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be square")
    if not torch.allclose(distances, distances.T, atol=1e-5, rtol=1e-5):
        raise ValueError("distance_matrix must be symmetric")
    if torch.any(distances < 0) or not torch.all(torch.isfinite(distances)):
        raise ValueError("distance_matrix must contain finite non-negative values")

    upper_triangle = distances[torch.triu_indices(distances.shape[0], distances.shape[1], offset=1).unbind()]
    positive_distances = upper_triangle[upper_triangle > 0]
    if positive_distances.numel() == 0:
        raise ValueError("distance_matrix must contain at least one positive distance")
    scale = torch.quantile(positive_distances, quantile)
    normalized = distances / scale
    diagonal_mask = torch.eye(normalized.shape[0], device=normalized.device, dtype=torch.bool)
    return normalized.masked_fill(diagonal_mask, 0.0)


def compute_normalized_persistence_diagrams(
    distance_matrix: torch.Tensor | np.ndarray,
    *,
    maxdim: int = 2,
    coeffs: Sequence[int] | None = None,
    scale_quantile: float = 0.9,
) -> dict[int, list[np.ndarray]]:
    """Compute Vietoris--Rips persistence after robust metric normalization."""

    normalized_distances = normalize_distance_matrix(distance_matrix, quantile=scale_quantile)
    return compute_persistence_diagrams(
        normalized_distances,
        maxdim=maxdim,
        coeffs=coeffs,
        distance_matrix=True,
    )


def compute_persistence_diagrams(
    data: torch.Tensor | np.ndarray,
    maxdim: int = 2,
    coeffs: Sequence[int] | None = None,
    distance_matrix: bool = False,
) -> dict[int, list[np.ndarray]]:
    coeff_list = list(coeffs) if coeffs is not None else [2, 3]
    if isinstance(data, torch.Tensor):
        data_array = data.detach().cpu().numpy()
    else:
        data_array = np.asarray(data)
    diagrams = {}
    for coeff in coeff_list:
        diagram = ripser(data_array, maxdim=maxdim, coeff=coeff, distance_matrix=distance_matrix)["dgms"]
        diagrams[coeff] = diagram
    return diagrams


def compute_pairwise_bottlenecks(
    original_diagrams: dict[int, list[np.ndarray]],
    reconstructed_diagrams: dict[int, list[np.ndarray]],
) -> dict[int, np.ndarray]:
    bottlenecks = {}
    for coeff in original_diagrams:
        bottlenecks[coeff] = np.array(
            [bottleneck(original_diagrams[coeff][i], reconstructed_diagrams[coeff][i]) for i in range(len(original_diagrams[coeff]))]
        )
    return bottlenecks


def compute_topology_bottlenecks(
    original_diagrams: dict[int, list[np.ndarray]],
    latent_diagrams: dict[int, list[np.ndarray]],
    *,
    homology_dimensions: Sequence[int] = (1, 2),
) -> tuple[dict[int, dict[int, float]], float]:
    """Compare topology-bearing homology groups and return their mean score.

    H0 is intentionally omitted by default: connected finite point clouds have
    a large collection of sampling-dependent H0 bars, while the Klein-bottle
    signature relevant to the manuscript is in H1 and H2 and changes with the
    coefficient field.
    """

    if set(original_diagrams) != set(latent_diagrams):
        raise ValueError("original and latent diagrams must use the same coefficient fields")

    distances: dict[int, dict[int, float]] = {}
    flattened: list[float] = []
    for coeff in original_diagrams:
        distances[coeff] = {}
        for dimension in homology_dimensions:
            if dimension >= len(original_diagrams[coeff]) or dimension >= len(latent_diagrams[coeff]):
                raise ValueError(f"homology dimension {dimension} is missing for coefficient field {coeff}")
            distance = float(bottleneck(original_diagrams[coeff][dimension], latent_diagrams[coeff][dimension]))
            distances[coeff][dimension] = distance
            flattened.append(distance)

    if not flattened:
        raise ValueError("homology_dimensions must not be empty")
    return distances, float(np.mean(flattened))


def plot_persistence_diagram(
    diagram: list[np.ndarray], title: str = "Persistence Diagram", ax: Axes | None = None
) -> Axes | tuple[Figure, Axes]:
    return_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        return_fig = True
    plot_diagrams(diagram, ax=ax)

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    if return_fig:
        return fig, ax
    else:
        return ax


def plot_persistence_diagram_detailed(diagram: list[np.ndarray], title: str, ax: Axes, show_ylabels: bool = True):
    """
    Plots a detailed, black-and-white persistence diagram on a given Axes object.

    Args:
        diagram: A list of numpy arrays [H0, H1, H2, ...].
        title: The title for the subplot.
        ax: The Matplotlib Axes object to draw on.
        show_ylabels: Whether to show the y-axis labels and ticks.
    """
    # Plot H1 and H2 with specific markers
    if len(diagram) > 1 and diagram[1].size > 0:
        ax.scatter(diagram[1][:, 0], diagram[1][:, 1], marker=".", color="black", s=100, label="$H_1$")
    if len(diagram) > 2 and diagram[2].size > 0:
        ax.scatter(diagram[2][:, 0], diagram[2][:, 1], marker="+", color="black", s=350, label="$H_2$")

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    max_limit = max(xlim[1], ylim[1])
    # Calculate the total range of the plot's data
    data_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0])

    # Set the radius to be a small fraction of the data range (e.g., 3%)
    dynamic_radius = data_range * 0.03

    # Draw the diagonal line based on this dynamic limit
    ax.plot([0, max_limit], [0, max_limit], color="black", alpha=0.3)

    # Circle the 2nd most persistent H1 point
    if len(diagram) > 1 and diagram[1].shape[0] > 1:
        lifetimes_h1 = diagram[1][:, 1] - diagram[1][:, 0]
        idx_h1 = np.argsort(lifetimes_h1)[-2]
        point_h1 = diagram[1][idx_h1, :]
        ax.add_patch(Circle((point_h1[0], point_h1[1]), radius=dynamic_radius, edgecolor="black", fill=False))

    # Circle the most persistent H2 point
    if len(diagram) > 2 and diagram[2].shape[0] > 0:
        lifetimes_h2 = diagram[2][:, 1] - diagram[2][:, 0]
        idx_h2 = np.argsort(lifetimes_h2)[-1]
        point_h2 = diagram[2][idx_h2, :]
        ax.add_patch(Circle((point_h2[0], point_h2[1]), radius=dynamic_radius, edgecolor="black", fill=False))

    # --- Formatting ---
    ax.set_aspect(1)
    ax.set_title(title, fontsize=30)
    ax.set_xlabel("Birth time", fontsize=17)
    ax.tick_params(axis="both", which="major", labelsize=15)

    if show_ylabels:
        ax.set_ylabel("Death time", fontsize=17)
    else:
        ax.set_yticks([])

    ax.legend(loc="lower right", prop={"size": 17})
    return ax
