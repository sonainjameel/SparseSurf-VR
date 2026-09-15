import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class OpticalAppearanceModel(nn.Module):
    """
    Per-Gaussian optical appearance parameters.

    Geometry remains in GaussianModel.

    This model adds:
      - a high-frequency spherical-Gaussian reflection lobe
      - optical transmission independent of geometric occupancy
    """

    def __init__(self, n_points, device="cuda"):
        super().__init__()

        # Small positive specular amplitude at initialization.
        self.specular_amplitude_raw = nn.Parameter(
            torch.full((n_points, 3), -6.0, device=device)
        )

        # Preferred incoming/reflection direction.
        axis = torch.randn(n_points, 3, device=device)
        axis = F.normalize(axis, dim=-1)
        self.specular_axis_raw = nn.Parameter(axis)

        # Positive concentration of the spherical Gaussian.
        # Starts broad enough to optimize safely, but can sharpen.
        self.specular_sharpness_raw = nn.Parameter(
            torch.full((n_points, 1), 3.0, device=device)
        )

        # Nearly opaque initially:
        # sigmoid(-6) ~= 0.0025 transmission.
        self.transmission_logit = nn.Parameter(
            torch.full((n_points, 1), -6.0, device=device)
        )

        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": [self.specular_amplitude_raw],
                    "lr": 2.5e-3,
                    "name": "optical_spec_amp"
                },
                {
                    "params": [self.specular_axis_raw],
                    "lr": 1.0e-3,
                    "name": "optical_spec_axis"
                },
                {
                    "params": [self.specular_sharpness_raw],
                    "lr": 5.0e-4,
                    "name": "optical_spec_sharpness"
                },
                {
                    "params": [self.transmission_logit],
                    "lr": 1.0e-3,
                    "name": "optical_transmission"
                },
            ],
            lr=0.0,
            eps=1e-15
        )

    @property
    def specular_amplitude(self):
        return F.softplus(self.specular_amplitude_raw)

    @property
    def specular_axis(self):
        return F.normalize(self.specular_axis_raw, dim=-1)

    @property
    def specular_sharpness(self):
        # Positive and unbounded, allowing narrow angular lobes.
        return 1.0 + F.softplus(self.specular_sharpness_raw)

    @property
    def transmission(self):
        return torch.sigmoid(self.transmission_logit)

    def regularization(self):
        """
        Sparse optical prior:
        most indoor Gaussians should remain opaque and should not need
        a strong high-frequency reflection lobe.
        """
        spec_prior = self.specular_amplitude.mean()
        transmission_prior = self.transmission.mean()

        return (
            1.0e-5 * spec_prior
            + 1.0e-4 * transmission_prior
        )

    def save(self, model_path, iteration):
        out_dir = os.path.join(
            model_path,
            "optical_model",
            "iteration_{}".format(iteration)
        )
        os.makedirs(out_dir, exist_ok=True)

        torch.save(
            {
                "state_dict": self.state_dict(),
                "n_points": self.transmission_logit.shape[0],
            },
            os.path.join(out_dir, "optical.pth")
        )

    @classmethod
    def load(cls, path, device="cuda"):
        payload = torch.load(
            path,
            map_location=device,
            weights_only=False
        )

        model = cls(payload["n_points"], device=device)
        model.load_state_dict(payload["state_dict"])
        return model
