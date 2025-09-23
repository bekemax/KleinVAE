import torch

from typing import Callable, Optional, Union


def projection_onto_a_line(theta: torch.Tensor) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Creates a function that projects points onto a line.
    Args:
        theta: Angle tensor of any shape
    Returns:
        Function that takes x,y coordinates and projects them onto the line
    """

    def project(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * torch.cos(theta) + y * torch.sin(theta)

    return project


def chebyshev_2(t: torch.Tensor) -> torch.Tensor:
    """
    Chebyshev polynomial of the 2nd degree.

    Args:
        t: Input tensor of any shape

    Returns:
        Tensor of same shape as input with Chebyshev polynomial applied
    """
    return 2 * t.pow(2) - 1


def klein_filter(theta_1: torch.Tensor, theta_2: torch.Tensor) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Klein filter for given angles theta_1 and theta_2.

    Args:
        theta_1: First angle tensor
        theta_2: Second angle tensor

    Returns:
        Function that takes x,y coordinates and applies the Klein filter
    """
    t = projection_onto_a_line(theta_1)

    def filter_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sin(theta_2) * t(x, y) + torch.cos(theta_2) * chebyshev_2(t(x, y))

    return filter_fn


def generate_klein_filter_matrix(
    theta_1: Union[float, torch.Tensor],
    theta_2: Union[float, torch.Tensor],
    size: int = 3,
    midpoint: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate a Klein filter matrix of given size for specified angles.

    Args:
        theta_1: First angle tensor
        theta_2: Second angle tensor
        size: Size of the matrix
        midpoint: If True, use midpoints for the grid; otherwise, use edges
        device: PyTorch device to place tensors on. If None, uses theta_1's device

    Returns:
        Tensor of shape [size, size] containing the Klein filter values
    """

    if isinstance(theta_1, float):
        theta_1 = torch.tensor(theta_1, device=device)
    if isinstance(theta_2, float):
        theta_2 = torch.tensor(theta_2, device=device)

    if device is None:
        device = theta_1.device

    f = klein_filter(theta_1, theta_2)

    if midpoint:
        coords = torch.linspace(-1 + 1 / size, 1 - 1 / size, size, device=device)
    else:
        coords = torch.linspace(-1, 1, size, device=device)

    X, Y = torch.meshgrid(coords, coords, indexing="ij")
    return f(X, Y)
