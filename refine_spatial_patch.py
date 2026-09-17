import os
import random
import shutil
import torch

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render, GaussianModel
from scene import Scene
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim


def main():
    parser = ArgumentParser(
        description="Post-hoc spatial Gaussian appearance refinement"
    )

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", type=int, default=7000)
    parser.add_argument("--refine_iterations", type=int, required=True)
    parser.add_argument("--patch_lr", type=float, default=0.0025)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)

    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    baseline_path = os.path.abspath(dataset.model_path)
    output_path = os.path.abspath(args.output_path)

    assert baseline_path != output_path, (
        "output_path must be different from the baseline model path"
    )

    print("BASELINE_MODEL", baseline_path)
    print("OUTPUT_PATH", output_path)
    print("LOAD_ITERATION", args.iteration)
    print("REFINE_ITERATIONS", args.refine_iterations)
    print("PATCH_LR", args.patch_lr)
    print("LAMBDA_DSSIM", args.lambda_dssim)

    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
    )

    train_cameras = scene.getTrainCameras().copy()

    assert len(train_cameras) > 0

    print("TRAIN_CAMERAS", len(train_cameras))
    print("GAUSSIANS", gaussians.get_xyz.shape[0])

    patch = gaussians.get_spatial_patch

    assert patch.ndim == 3
    assert patch.shape[1:] == (4, 3)
    assert patch.shape[0] == gaussians.get_xyz.shape[0]

    # Freeze every existing Gaussian representation component.
    frozen_names = [
        "_xyz",
        "_knn_f",
        "_features_dc",
        "_features_rest",
        "_surf_feat",
        "_opacity",
        "_scaling",
        "_rotation",
    ]

    for name in frozen_names:
        value = getattr(gaussians, name, None)
        if isinstance(value, torch.Tensor):
            value.requires_grad_(False)

    patch.requires_grad_(True)

    # The post-hoc stage optimizes only the spatial appearance patch.
    optimizer = torch.optim.Adam(
        [patch],
        lr=args.patch_lr,
        eps=1e-15,
    )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]

    background = torch.tensor(
        bg_color,
        dtype=torch.float32,
        device="cuda",
    )

    print("PATCH_INITIAL_MAX_ABS", patch.detach().abs().max().item())

    # Deterministic cycling uses only real training views.
    for refine_iter in range(1, args.refine_iterations + 1):
        view = train_cameras[
            (refine_iter - 1) % len(train_cameras)
        ]

        render_pkg = render(
            view,
            gaussians,
            pipe,
            background,
        )

        image = render_pkg["render"]
        gt_image = view.original_image[0:3, :, :].cuda()

        Ll1 = l1_loss(image, gt_image)

        loss = (
            (1.0 - args.lambda_dssim) * Ll1
            + args.lambda_dssim * (1.0 - ssim(image, gt_image))
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss at refinement iteration "
                + str(refine_iter)
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad = patch.grad

        if grad is None:
            raise RuntimeError(
                "Spatial patch gradient is missing at iteration "
                + str(refine_iter)
            )

        if not torch.isfinite(grad).all():
            raise RuntimeError(
                "Non-finite spatial patch gradient at iteration "
                + str(refine_iter)
            )

        optimizer.step()

        if (
            refine_iter == 1
            or refine_iter % 10 == 0
            or refine_iter == args.refine_iterations
        ):
            print(
                "REFINE",
                refine_iter,
                "VIEW",
                view.image_name,
                "LOSS",
                float(loss.detach()),
                "PATCH_MAX_ABS",
                float(patch.detach().abs().max()),
                "GRAD_MAX_ABS",
                float(grad.detach().abs().max()),
            )

    save_dir = os.path.join(
        output_path,
        "point_cloud",
        "iteration_" + str(args.refine_iterations),
    )

    os.makedirs(save_dir, exist_ok=True)

    os.makedirs(output_path, exist_ok=True)

    baseline_cfg = os.path.join(
        baseline_path,
        "cfg_args",
    )
    output_cfg = os.path.join(
        output_path,
        "cfg_args",
    )

    if os.path.isfile(baseline_cfg):
        shutil.copy2(
            baseline_cfg,
            output_cfg,
        )
        print("COPIED_CFG", output_cfg)

    save_path = os.path.join(
        save_dir,
        "point_cloud.ply",
    )

    gaussians.save_ply(save_path)

    print("SAVED", save_path)
    print("PATCH_FINAL_MAX_ABS", patch.detach().abs().max().item())
    print("SPATIAL_PATCH_REFINEMENT_OK")


if __name__ == "__main__":
    main()
