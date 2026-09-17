import torch
import torch.nn.functional as F

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state


def main():
    parser = ArgumentParser(description="Spatial patch gradient smoke test")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
    )

    patch = gaussians.get_spatial_patch

    assert patch.ndim == 3
    assert patch.shape[1] == 4
    assert patch.shape[2] == 3
    assert patch.shape[0] == gaussians.get_xyz.shape[0]

    print("GAUSSIANS", gaussians.get_xyz.shape[0])
    print("PATCH_SHAPE", tuple(patch.shape))
    print("PATCH_REQUIRES_GRAD", patch.requires_grad)
    print("PATCH_INITIAL_MAX_ABS", patch.detach().abs().max().item())

    # Freeze the existing Gaussian representation.
    # The smoke test is only checking gradients for the new spatial patch.
    frozen_names = [
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_opacity",
        "_scaling",
        "_rotation",
        "_surf_feat",
    ]

    for name in frozen_names:
        tensor = getattr(gaussians, name, None)
        if isinstance(tensor, torch.Tensor):
            tensor.requires_grad_(False)

    assert patch.requires_grad

    view = scene.getTrainCameras()[0]

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(
        bg_color,
        dtype=torch.float32,
        device="cuda",
    )

    if patch.grad is not None:
        patch.grad.zero_()

    render_pkg = render(
        view,
        gaussians,
        pipe,
        background,
    )

    rendered = render_pkg["render"]
    gt = view.original_image[0:3, :, :]

    loss = F.l1_loss(rendered, gt)

    print("VIEW", view.image_name)
    print("RENDER_SHAPE", tuple(rendered.shape))
    print("LOSS", loss.item())

    loss.backward()

    grad = patch.grad

    assert grad is not None
    assert grad.shape == patch.shape

    finite = torch.isfinite(grad)
    nonzero = grad != 0

    print("GRAD_FINITE_ALL", finite.all().item())
    print("GRAD_NONZERO_COUNT", nonzero.sum().item())
    print("GRAD_TOTAL_COUNT", grad.numel())
    print("GRAD_MEAN_ABS", grad.abs().mean().item())
    print("GRAD_MAX_ABS", grad.abs().max().item())

    assert finite.all()
    assert nonzero.any()

    print("SPATIAL_PATCH_GRADIENT_SMOKE_OK")


if __name__ == "__main__":
    main()
