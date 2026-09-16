import torch
import torch.nn.functional as F


@torch.no_grad()
def build_detail_densification_route(
    gaussians,
    viewpoint_cam,
    gt_image,
    rendered_image,
    radii,
    visibility_filter,
    valid_indices,
    ncc_error,
    ncc_threshold,
    patch_size,
    height,
    width,
):
    """
    Route late-stage densification according to the cause of image residuals.

    route < 1:
        cross-view inconsistent evidence, such as unstable reflection,
        occlusion or correspondence ambiguity.

    route ~= 1:
        ordinary SparseSurf behaviour.

    route > 1:
        cross-view-consistent high-frequency residual whose projected
        Gaussian footprint is too large to represent the required detail.

    Returned tensor is per Gaussian and bounded to [0.5, 2.0].
    """

    device = rendered_image.device
    dtype = rendered_image.dtype

    # ------------------------------------------------------------
    # 1. SparseSurf multi-view consistency -> dense image map
    # ------------------------------------------------------------
    consistency_samples = torch.clamp(
        1.0 - ncc_error.detach() / max(float(ncc_threshold), 1e-6),
        0.0,
        1.0,
    )

    consistency_map = torch.zeros(
        (1, 1, height, width),
        device=device,
        dtype=dtype,
    )

    consistency_map.view(-1)[valid_indices] = consistency_samples

    consistency_map = F.max_pool2d(
        consistency_map,
        kernel_size=2 * patch_size + 1,
        stride=1,
        padding=patch_size,
    )

    # ------------------------------------------------------------
    # 2. Residual spatial frequency
    #
    # For e(x) = A sin(2*pi*f*x):
    #
    # |grad e| / (2*pi*A) approximates local frequency.
    # ------------------------------------------------------------
    residual = (
        gt_image.detach() - rendered_image.detach()
    ).abs().mean(dim=0, keepdim=True).unsqueeze(0)

    dx = torch.zeros_like(residual)
    dy = torch.zeros_like(residual)

    dx[:, :, :, 1:-1] = 0.5 * (
        residual[:, :, :, 2:] - residual[:, :, :, :-2]
    )

    dy[:, :, 1:-1, :] = 0.5 * (
        residual[:, :, 2:, :] - residual[:, :, :-2, :]
    )

    grad_mag = torch.sqrt(dx * dx + dy * dy + 1e-12)

    local_amp = torch.sqrt(
        F.avg_pool2d(
            residual * residual,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        + 1e-12
    )

    local_grad = F.avg_pool2d(
        grad_mag,
        kernel_size=7,
        stride=1,
        padding=3,
    )

    local_frequency = local_grad / (
        2.0 * torch.pi * (local_amp + 1e-6)
    )

    # Ignore tiny numerical residuals.
    residual_scale = residual.median() + 1e-6

    amplitude_confidence = local_amp / (
        local_amp + residual_scale
    )

    # ------------------------------------------------------------
    # 3. Project Gaussian centres into the current image
    # ------------------------------------------------------------
    xyz = gaussians.get_xyz.detach()

    pts_cam = (
        xyz @ viewpoint_cam.world_view_transform[:3, :3]
        + viewpoint_cam.world_view_transform[3, :3]
    )

    z = pts_cam[:, 2]

    px = (
        pts_cam[:, 0] * viewpoint_cam.Fx
        / z.clamp(min=1e-6)
        + viewpoint_cam.Cx
    )

    py = (
        pts_cam[:, 1] * viewpoint_cam.Fy
        / z.clamp(min=1e-6)
        + viewpoint_cam.Cy
    )

    ix = torch.round(px).long()
    iy = torch.round(py).long()

    valid = (
        visibility_filter
        & (z > 1e-6)
        & (ix >= 0)
        & (ix < width)
        & (iy >= 0)
        & (iy < height)
    )

    # Default: exactly baseline SparseSurf.
    route = torch.ones(
        xyz.shape[0],
        device=device,
        dtype=dtype,
    )

    if not valid.any():
        return route

    freq_g = local_frequency[
        0, 0, iy[valid], ix[valid]
    ]

    consistency_g = consistency_map[
        0, 0, iy[valid], ix[valid]
    ]

    amplitude_g = amplitude_confidence[
        0, 0, iy[valid], ix[valid]
    ]

    radius_g = radii[valid].float()

    # ------------------------------------------------------------
    # 4. Sampling-theoretic detail violation
    #
    # wavelength = 1/f
    #
    # Require approximately:
    #     Gaussian diameter <= half wavelength
    #
    # hence:
    #     4 * radius * frequency <= 1
    # ------------------------------------------------------------
    frequency_violation = torch.relu(
        4.0 * radius_g * freq_g - 1.0
    )

    stable_detail = (
        consistency_g
        * amplitude_g
        * frequency_violation
    )

    # ------------------------------------------------------------
    # 5. Causal densification route
    #
    # inconsistent residual:
    #     suppress toward 0.5
    #
    # stable ordinary surface:
    #     approximately 1
    #
    # stable unresolved detail:
    #     boost toward 2
    # ------------------------------------------------------------
    route_g = (
        1.0
        - 0.5 * (1.0 - consistency_g)
        + torch.tanh(stable_detail)
    )

    route[valid] = torch.clamp(
        route_g,
        min=0.5,
        max=2.0,
    )

    return route
