import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ripser import ripser
from persim import bottleneck, plot_diagrams

from typing import Callable, Dict, List, Optional, Tuple, Union


def project_to_torus(points: torch.Tensor, stack: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """
    Given (u, v) in R^2, project each pair (u, v) onto the fundamental domain
    [0, 2pi) x [0, 2pi) of the torus by wrapping u and v according to the identifications:
        - (theta1, theta2) ~ (theta1 + 2pi k, theta2 + 2pi l) for k,l in Z
    Args:
        points: Tensor of shape (..., 2) representing points in R^2
        stack: If True, return a stacked tensor of shape (..., 2). If False, return a tuple of tensors (u_mod, v_mod).
    Returns:
        If stack is True, a tensor of shape (..., 2) representing points on the torus.
        If stack is False, a tuple of tensors (u_mod, v_mod) each of shape (...,).
    """
    u, v = points[..., 0], points[..., 1]
    u_mod = torch.remainder(u, 2)
    v_mod = torch.remainder(v, 1)
    if stack:
        return torch.stack([u_mod, v_mod], dim=-1)
    else:
        return u_mod, v_mod


def project_to_klein(points: torch.Tensor, eps: float = 1e-6):
    """
    Given (u, v) in R^2, project each pair (u, v) onto the fundamental domain
    [0, 2pi) x [0, pi) of the Klein bottle by wrapping u and v according to the identifications:
        - (theta1, theta2) ~ (theta1 + 2pi k, theta2 + 2pi l)    for k,l in Z (torus gluing)
        - (theta1, theta2) ~ (theta1 - pi, -theta1 (mod 2pi))              (the “half‐twist” that makes K non-orientable)
    """

    u_mod, v_mod = project_to_torus(points)

    mask_twist = u_mod >= 1
    if mask_twist.any():
        u_mod[mask_twist] = u_mod[mask_twist] - 1
        v_mod[mask_twist] = 1 - v_mod[mask_twist]  # torch.remainder(-v_mod[mask_twist], 1)  # 1 - v_mod[mask_twist]

    return torch.stack([u_mod, v_mod], dim=-1)


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
    xs_torus = torch.stack([xs, xs + torch.pi], dim=-1)
    ys_torus = torch.stack([ys, torch.remainder(-ys, 2 * torch.pi)], dim=-1)

    # print(torch.stack([xs_torus, ys_torus], dim=-1).shape)

    return torch.stack([xs_torus, ys_torus], dim=-1).squeeze(1)


def preimage_of_torus_grid(points, grid_size=1):
    """
    Given points in the torus, return all preimages in a grid around each point.
    Args:
        points: Tensor of shape (..., 2) representing points in the torus.
        grid_size: Size of the grid to generate preimages around each point.
    Returns:
        Tensor of shape (..., (2 * grid_size + 1) ** 2, 2) representing preimages in the torus.
    """
    xs, ys = points[..., 0], points[..., 1]
    shifts = range(-grid_size, grid_size + 1)
    all_preimages = [torch.stack([xs + 2 * torch.pi * i, ys + 2 * torch.pi * j], dim=-1) for i in shifts for j in shifts]
    return torch.cat(all_preimages, dim=1)


def preimage_of_torus_recursive(points: torch.Tensor, criterion: Callable[[torch.Tensor], bool]):
    """
    Given points in the torus, recursively find preimages until the criterion is satisfied.
    The criterion is a function that takes a tensor of points and returns True if the points satisfy the condition.
    """
    grid = 2 * torch.pi * torch.linspace(-1, 1, 3)
    meshgrid = torch.meshgrid(grid, grid, indexing="ij")
    meshgrid_points = torch.column_stack([meshgrid[0].flatten(), meshgrid[1].flatten()])
    preimages = torch.Tensor([meshgrid_points + point for point in points]).reshape(-1, 2)
    print(f"Preimages: {preimages}")
    if criterion(preimages):
        return preimages
        return preimage_of_torus_recursive(preimages, criterion)


def compute_persistence_diagrams(data: torch.Tensor, maxdim: int = 2, coeffs: List[int] = [2, 3]) -> Dict[int, List[np.ndarray]]:
    diagrams = {}
    for coeff in coeffs:
        diagram = ripser(data, maxdim=maxdim, coeff=coeff)["dgms"]
        diagrams[coeff] = diagram
    return diagrams


def compute_pairwise_bottlenecks(
    original_diagrams: Dict[int, List[np.ndarray]], reconstructed_diagrams: Dict[int, List[float]]
) -> Dict[int, np.ndarray]:
    bottlenecks = {}
    for coeff in original_diagrams.keys():
        bottlenecks[coeff] = np.array(
            [bottleneck(original_diagrams[coeff][i], reconstructed_diagrams[coeff][i]) for i in range(len(original_diagrams[coeff]))]
        )
    return bottlenecks


def plot_persistence_diagram(
    diagram: List[np.ndarray], title: str = "Persistence Diagram", ax: Optional[Axes] = None
) -> Union[Axes, Tuple[Figure, Axes]]:
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


def plot_persistence_diagram_detailed(diagram: List[np.ndarray], title: str, ax: Axes, show_ylabels: bool = True):
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
