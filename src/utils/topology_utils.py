import torch

from typing import Callable, Tuple, Union


def project_to_torus(points: torch.Tensor, stack: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
    else:
        return preimage_of_torus_recursive(preimages, criterion)
