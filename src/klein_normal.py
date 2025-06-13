import torch
from torch.distributions import Normal, constraints

from math import pi


class KleinNormal(Normal):
    """
    A wrapped 2D Normal on the Klein bottle K, using (θ1, θ2) in the fundamental
    rectangle [0, 2π) × [0, π) with the identifications:
       (θ1, θ2) ~ (θ1 + 2πk, θ2 + 2πl)    for k,l ∈ Z  (usual torus gluing)
       (θ1, θ2) ~ (θ1 + π, -θ2)         (the “half‐twist” that makes K non-orientable)

    We override only:
      - .rsample()   → to draw (u,v) ∼ N(loc, scale) in ℝ² and then project to (φ1, φ2) ∈ [0,2π)×[0,π).
      - .log_prob()  → to take any (θ1, θ2), re-project it to the canonical (φ1, φ2) in [0,2π)×[0,π), and
                       then call Normal.log_prob((φ1, φ2)) in the covering space.

    Everything else (entropy, expand, etc.) is inherited from torch.distributions.Normal.
    """

    support = constraints.real  # the parent Normal thinks in ℝ²; we reinterpret internally.

    def __init__(self, loc, scale, validate_args=False):
        """
        loc:   Tensor of shape [2], giving the mean (u₀,v₀) ∈ ℝ² of the covering Gaussian.
        scale: Tensor of shape [2], giving the std ‐dev in each coordinate of that covering Gaussian.
        """
        super().__init__(loc, scale, validate_args=validate_args)

    def _project_to_klein(self, u, v, eps: float = 1e-6):
        """
        Given u, v ∈ ℝ^*, project each pair (u, v) onto the fundamental domain
        [0, 2π) × [0, π) of the Klein bottle by wrapping u and v according to the identifications:
            - (θ1, θ2) ~ (θ1 + 2πk, θ2 + 2πl) for k,l ∈ Z    (usual torus gluing)
            - (θ1, θ2) ~ (θ1 + π, -θ2)                       (the “half‐twist” that makes K non-orientable)
        """

        u_mod = torch.remainder(u, 2 * pi)
        v_mod = torch.remainder(v, 2 * pi)

        mask_twist = v_mod >= pi - eps
        if mask_twist.any():
            u_mod[mask_twist] = u_mod[mask_twist] - pi
            v_mod[mask_twist] = torch.remainder(-v_mod[mask_twist], 2 * pi)

        return u_mod, v_mod

    def rsample(self, sample_shape=torch.Size()):
        """
        1) Draw (u,v) ~ N(loc, scale) in ℝ² via super().rsample().
        2) Project (u,v) using the _project_to_klein() rule above.
        3) Return a tensor of shape sample_shape + [2].
        """
        uv = super().rsample(sample_shape)  # shape = [..., 2]
        u, v = uv[..., 0], uv[..., 1]  # each shape = [...]

        u_klein, v_klein = self._project_to_klein(u, v)

        return torch.stack([u_klein, v_klein], dim=-1)

    #! THIS FUNCTION IS WRONG: REDO IT ON MY OWN
    def log_prob(self, theta):
        """
        Given any point theta = (θ1_in, θ2_in) ∈ ℝ² (or already in [0,2π)×[0,π)):
        1) First fold θ1_in, θ2_in into their torus‐cover:
              u_mod  = θ1_in % (2π),  v_mod = θ2_in % (2π).
        2) Then apply exactly the same “if v_mod < π vs ≥ π” logic to recover
           (φ1, φ2) ∈ [0,2π)×[0,π), the unique canonical rep on K.
        3) Call parent Normal.log_prob((φ1, φ2)) → returns two partial log‐densities [..., 2].
        4) Sum them to get a scalar log‐density on the Klein bottle → return shape [...].

        This ensures log_prob is consistent across all equivalent lifts.
        """
        theta1_in = theta[..., 0]
        theta2_in = theta[..., 1]

        two_pi = 2.0 * pi

        # a) Reduce each input coord into [0, 2π)
        u_mod = torch.remainder(theta1_in, two_pi)  # [...,]
        v_mod = torch.remainder(theta2_in, two_pi)  # [...,]

        # b) Allocate tensors for φ1, φ2
        phi1 = torch.empty_like(u_mod)
        phi2 = torch.empty_like(v_mod)

        # c) Apply same “bottom vs top” logic
        mask_bottom = v_mod < pi
        mask_top = v_mod >= pi

        # bottom: φ1 = u_mod, φ2 = v_mod
        if mask_bottom.any():
            phi1[mask_bottom] = u_mod[mask_bottom]
            phi2[mask_bottom] = v_mod[mask_bottom]

        # top: φ1 = (u_mod + π) % (2π), φ2 = 2π - v_mod
        if mask_top.any():
            phi1[mask_top] = torch.remainder(u_mod[mask_top] + pi, two_pi)
            phi2[mask_top] = two_pi - v_mod[mask_top]

        # d) Stack back into [..., 2] and delegate to Normal.log_prob
        uv0 = torch.stack([phi1, phi2], dim=-1)  # shape [..., 2]
        lp = super().log_prob(uv0)  # shape [..., 2]
        return lp.sum(dim=-1)  # summation → shape [...]

    @property
    def mean(self):
        """
        Wrap the base Normal’s mean = self.loc into [0,2π)×[0,π).
        This is optional, but convenient if any code inspects `dist.mean`.
        """
        u0, v0 = self.loc[0], self.loc[1]
        phi1, phi2 = self._project_to_klein(u0.unsqueeze(0), v0.unsqueeze(0))
        # phi1, phi2 each shape [1]; return a length-2 tensor
        return torch.cat([phi1, phi2], dim=0)
