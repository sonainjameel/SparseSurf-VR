#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
from scene import Scene
import os
from tqdm import tqdm
import numpy as np
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.optical_appearance_model import OpticalAppearanceModel
import cv2


def visualize_depth(depth):
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    if depth_max <= depth_min:
        depth_norm = np.zeros_like(depth, dtype=np.uint8)
    else:
        depth_norm = ((depth - depth_min) / (depth_max - depth_min) * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)



def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args):

    # ========================================================
    # OPTICAL MODEL LOAD
    # ========================================================
    # Iterations before 4500 are ordinary SparseSurf.
    # From 4500 onward the saved optical representation is
    # mandatory. Never silently fall back to baseline rendering.
    optical_model = None

    # Ablation branch: optical appearance is intentionally disabled.
    enable_optical_eval = False

    if enable_optical_eval and iteration >= 4500:
        optical_path = os.path.join(
            model_path,
            "optical_model",
            "iteration_{}".format(iteration),
            "optical.pth"
        )

        if not os.path.isfile(optical_path):
            raise FileNotFoundError(
                "Optical model required for iteration {} "
                "but was not found: {}".format(
                    iteration,
                    optical_path
                )
            )

        optical_model = OpticalAppearanceModel.load(
            optical_path,
            device=str(gaussians.get_xyz.device)
        )

        optical_model.eval()

        for parameter in optical_model.parameters():
            parameter.requires_grad_(False)

        n_geometry = gaussians.get_xyz.shape[0]
        n_optical = (
            optical_model
            .transmission_logit
            .shape[0]
        )

        if n_geometry != n_optical:
            raise RuntimeError(
                "Geometry/optical Gaussian count mismatch: "
                "geometry={}, optical={}".format(
                    n_geometry,
                    n_optical
                )
            )

        print(
            "[RENDER] Loaded optical appearance: "
            "{} Gaussians from {}".format(
                n_optical,
                optical_path
            )
        )

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)


    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background, optical_model=optical_model)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, view.image_name + '.png'))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

        if args.render_depth:
            depth_map = visualize_depth(render_pkg['plane_depth'][0].detach().cpu().numpy())
            cv2.imwrite(os.path.join(render_path, view.image_name + '_depth.png'), depth_map)


def render_sets(dataset : ModelParams, pipeline : PipelineParams, args):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        print(f"point number is {gaussians.get_xyz.shape[0]}")

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not args.skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)
        if not args.skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--fps", default=25, type=int)
    parser.add_argument("--render_depth", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), pipeline.extract(args), args)
