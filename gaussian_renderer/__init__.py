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
import matplotlib.pyplot as plt
import torch
import math
from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.app_model import AppModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import normal_from_depth_image

def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale], 
                                            intrinsic_matrix.to(depth.device), 
                                            extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, 
           app_model: AppModel=None, optical_model=None, return_plane = True, return_depth_normal = True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    return_dict = None

    # --------------------------------------------------------
    # Residual-Decomposed SparseSurf optical appearance
    # --------------------------------------------------------
    optical_colors = None
    visual_opacity = None

    if optical_model is not None:
        # Geometry is intentionally detached from RGB appearance.
        xyz_opt = pc.get_xyz.detach()

        to_camera = (
            viewpoint_camera.camera_center[None, :] - xyz_opt
        )
        to_camera = torch.nn.functional.normalize(
            to_camera,
            dim=-1
        )

        normal_opt = pc.get_normal(viewpoint_camera).detach()
        normal_opt = torch.nn.functional.normalize(
            normal_opt,
            dim=-1
        )

        # Orient normal toward the current camera.
        facing = (
            (normal_opt * to_camera).sum(dim=-1, keepdim=True)
            < 0
        )
        normal_opt = torch.where(
            facing,
            -normal_opt,
            normal_opt
        )

        # Reflection direction:
        # r = 2(n.v)n - v
        ndotv = (
            normal_opt * to_camera
        ).sum(dim=-1, keepdim=True)

        reflection_dir = (
            2.0 * ndotv * normal_opt - to_camera
        )

        reflection_dir = torch.nn.functional.normalize(
            reflection_dir,
            dim=-1
        )

        # Existing SparseSurf SH remains the stable/base appearance.
        shs_view = (
            pc.get_features
            .transpose(1, 2)
            .view(
                -1,
                3,
                (pc.max_sh_degree + 1) ** 2
            )
        )

        # Standard 3DGS SH convention: camera -> Gaussian.
        base_dir = (
            xyz_opt
            - viewpoint_camera.camera_center[None, :]
        )
        base_dir = torch.nn.functional.normalize(
            base_dir,
            dim=-1
        )

        base_rgb = (
            eval_sh(
                pc.active_sh_degree,
                shs_view,
                base_dir
            )
            + 0.5
        )
        base_rgb = torch.clamp(base_rgb, min=0.0)

        # High-frequency spherical-Gaussian reflection lobe.
        axis = optical_model.specular_axis
        sharpness = optical_model.specular_sharpness
        amplitude = optical_model.specular_amplitude

        angular_alignment = (
            reflection_dir * axis
        ).sum(dim=-1, keepdim=True)

        specular_lobe = torch.exp(
            sharpness
            * (angular_alignment - 1.0)
        )

        specular_rgb = amplitude * specular_lobe

        # ----------------------------------------------------
        # Geometry opacity != optical opacity
        # ----------------------------------------------------
        transmission = optical_model.transmission

        # Schlick Fresnel approximation.
        F0 = 0.04

        cos_theta = torch.clamp(
            torch.abs(ndotv),
            0.0,
            1.0
        )

        fresnel = (
            F0
            + (1.0 - F0)
            * torch.pow(1.0 - cos_theta, 5.0)
        )

        # Fraction of foreground radiance that blocks the
        # background.  For opaque surfaces transmission -> 0.
        optical_extinction = (
            1.0
            - transmission * (1.0 - fresnel)
        )

        # Geometry opacity is detached here:
        # RGB appearance must not change physical occupancy.
        visual_opacity = (
            pc.get_opacity.detach()
            * optical_extinction
        )

        # Premultiplied physical decomposition:
        #
        # opaque:
        #   diffuse + Fresnel reflection
        #
        # transmissive:
        #   reflection remains while background passes through
        optical_numerator = (
            (1.0 - transmission)
            * (1.0 - fresnel)
            * base_rgb
            + fresnel * specular_rgb
        )

        optical_colors = (
            optical_numerator
            / optical_extinction.clamp(min=1.0e-4)
        )
    raster_settings = PlaneGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            render_geo=return_plane,
            debug=pipe.debug
        )

    rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)

    if not return_plane:
        use_optical = optical_model is not None

        rendered_image, radii, out_observe, _, _ = rasterizer(
            means3D = means3D.detach() if use_optical else means3D,
            means2D = means2D,
            means2D_abs = means2D_abs,
            shs = None if use_optical else shs,
            colors_precomp = optical_colors if use_optical else colors_precomp,
            opacities = visual_opacity if use_optical else opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        return_dict =  {"render": rendered_image,
                        "viewspace_points": screenspace_points,
                        "viewspace_points_abs": screenspace_points_abs,
                        "visibility_filter" : radii > 0,
                        "radii": radii,
                        "out_observe": out_observe}
        if app_model is not None and pc.use_app:
            appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid, device='cuda')]
            app_image = torch.exp(appear_ab[0]) * rendered_image + appear_ab[1]
            return_dict.update({"app_image": app_image})
        return return_dict

    global_normal = pc.get_normal(viewpoint_camera)
    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3]
    pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,:3] + viewpoint_camera.world_view_transform[3,:3]
    depth_z = pts_in_cam[:, 2]
    local_distance = (local_normal * pts_in_cam).sum(-1).abs()
    input_all_map = torch.zeros((means3D.shape[0], 13), device='cuda').float()
    input_all_map[:, :3] = local_normal
    input_all_map[:, 3] = 1.0
    input_all_map[:, 4] = local_distance
    feat_map = pc._surf_feat
    if feat_map is not None:
        input_all_map[:, 5:13] = feat_map


    rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        means2D_abs = means2D_abs,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        all_map = input_all_map,
        cov3D_precomp = cov3D_precomp)

    # Optical RGB is rendered separately so depth, normals,
    # visibility and geometric occupancy remain SparseSurf geometry.
    if optical_model is not None:
        optical_image, _, _, _, _ = rasterizer(
            means3D = means3D.detach(),
            means2D = means2D,
            means2D_abs = means2D_abs,
            shs = None,
            colors_precomp = optical_colors,
            opacities = visual_opacity,
            scales = scales.detach() if scales is not None else None,
            rotations = rotations.detach() if rotations is not None else None,
            cov3D_precomp = (
                cov3D_precomp.detach()
                if cov3D_precomp is not None
                else None
            )
        )

        rendered_image = optical_image

    rendered_normal = out_all_map[0:3]
    rendered_alpha = out_all_map[3:4, ]
    # print(f"rendered_alpha is {rendered_alpha.mean()}")
    rendered_distance = out_all_map[4:5, ]
    feature_map = out_all_map[5:13, ]
    
    return_dict =  {"render": rendered_image,
                    "viewspace_points": screenspace_points,
                    "viewspace_points_abs": screenspace_points_abs,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "out_observe": out_observe,
                    "rendered_normal": rendered_normal,
                    "plane_depth": plane_depth,
                    "rendered_distance": rendered_distance,
                    "rendered_alpha": rendered_alpha,
                    "feature_map": feature_map,
                    }
    
    if app_model is not None and pc.use_app:
        appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid, device='cuda')]
        app_image = torch.exp(appear_ab[0]) * rendered_image + appear_ab[1]
        return_dict.update({"app_image": app_image})   

    if return_depth_normal:
        depth_normal = render_normal(viewpoint_camera, plane_depth.squeeze()) * (rendered_alpha).detach()
        return_dict.update({"depth_normal": depth_normal})
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return return_dict