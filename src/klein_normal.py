from .utils import project_to_klein, preimage_of_klein, preimage_of_torus_grid

import torch
from torch.distributions import MultivariateNormal, constraints


class KleinConstraint(constraints.Constraint):
    def __init__(self):
        super().__init__()

    def check(self, value):
        # value is [..., 2]
        print(value.shape)
        theta1 = value[..., 0]
        theta2 = value[..., 1]
        return (theta1 >= 0) & (theta1 < torch.pi) & (theta2 >= 0) & (theta2 < 2 * torch.pi)


class KleinNormal(MultivariateNormal):
    """
    A wrapped 2D Normal on the Klein bottle K, using (theta1, theta2) in the fundamental rectangle [0, 2pi) x [0, pi)
    with the following identifications:
       (theta1, theta2) ~ (theta1 + 2pi k, theta2 + 2pi l)    for k,l in Z (torus gluing)
       (theta1, theta2) ~ (theta1 + pi, -theta1)              (the “half‐twist” that makes K non-orientable)

    We override only:
      - .rsample()   → to draw (u,v) ∼ N(loc, scale) in ℝ² and then project to a point on the Klein bottle
      - .log_prob()  → to take any (theta1, theta2) on the Klein bottle, find its preimages [(x,y)],
                       and for each preimage call Normal.log_prob((x, y)), then sum the results.
    """

    def __init__(self, loc, scale, grid_size: int = 3, validate_args=False):
        """
        loc:   Tensor of shape [2], giving the mean (u,v) in R^2 of the covering Gaussian.
        scale: Tensor of shape [2], giving the std ‐ dev in each coordinate of that covering Gaussian.
        """
        self.klein_support = KleinConstraint()

        if not isinstance(loc, torch.Tensor) or loc.shape[-1] != 2:
            print("loc must be a tensor of shape [2].")

        if not isinstance(scale, torch.Tensor) or scale.shape[-2:] != (2, 2):
            print("scale must be a tensor of shape [2].")

        self.grid_size = grid_size

        super().__init__(loc=loc, covariance_matrix=scale, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        """
        1) Draw (u,v) ~ N(loc, scale) in ℝ² via super().rsample().
        2) Project (u,v) using the _project_to_klein() rule above.
        3) Return a tensor of shape sample_shape + [2].
        """
        uv = super().rsample(sample_shape)  # shape = [..., 2]

        return project_to_klein(uv)

    # * working on the log_prob method - seems it works, need more tests (not just ipynb)
    def log_prob(self, value, eps: float = 1e-6):
        if self.klein_support.check(value).any() is False:
            raise ValueError("Value is not in the support of the Klein Normal distribution.")

        stop_criterion = self._create_stop_criterion(eps=eps)

        preimages_of_klein = preimage_of_klein(value)
        preimages_of_torus = preimage_of_torus_grid(preimages_of_klein, grid_size=self.grid_size)
        print(f"Preimages of Klein: {preimages_of_klein.shape}, Preimages of Torus: {preimages_of_torus.shape}")

        flat_preimages = preimages_of_torus.view(-1, preimages_of_torus.shape[-1])
        log_probs_all = super().log_prob(flat_preimages).view(preimages_of_torus.shape[:-1])
        return torch.logsumexp(log_probs_all, dim=-1)

    def _create_stop_criterion(self, eps: float = 1e-6):
        """
        Create a stop criterion function for `log_prob`.
        """

        def stop_criterion(points):
            return super().log_prob(points).sum() < eps

        return stop_criterion

    @property
    def mean(self):
        return project_to_klein(self.loc)
