# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

import json
import math
from dataclasses import dataclass
from typing import Literal, Optional
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

from utils.viewer_utils import OrbitCamera
from gaussian_renderer import GaussianModel, FlameGaussianModel
from gaussian_renderer import render
from mesh_renderer import NVDiffRenderer
import matplotlib.pyplot as plt
import cv2

@dataclass
class PipelineConfig:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False


@dataclass
class Config:
    pipeline: PipelineConfig
    """Pipeline settings for gaussian splatting rendering"""
    point_path: Optional[Path] = None
    """Path to the gaussian splatting file"""
    motion_path: Optional[Path] = None
    """Path to the motion file (npz)"""
    sh_degree: int = 3
    """Spherical Harmonics degree"""
    render_mode: Literal['rgb', 'depth', 'opacity'] = 'rgb'
    """NeRF rendering mode"""
    W: int = 960
    """GUI width"""
    H: int = 540
    """GUI height"""
    radius: float = 1
    """default GUI camera radius from center"""
    fovy: float = 20
    """default GUI camera fovy"""
    background_color: tuple[float] = (1., 1., 1.)
    """default GUI background color"""
    save_folder: Path = Path("./viewer_output")
    """default saving folder"""
    fps: int = 25
    """default fps for recording"""
    keyframe_interval: int = 1
    """default keyframe interval"""
    ref_json: Optional[Path] = None
    """Path to the reference json file. We use this file to complement the exported trajectory json file."""
    demo_mode: bool = False
    """The UI will be simplified in demo mode."""

class GaussianSplattingViewer:
    def __init__(self):
        self.cam = OrbitCamera(960, 540, r=1, fovy=20, convention="opencv")

        # rendering settings
        self.scaling_modifier: float = 1
        self.num_timesteps = 1
        self.timestep = 0
        self.show_spatting = True
        self.show_mesh = False
        self.mesh_color = torch.tensor([1, 1, 1, 0.5])
        print("Initializing mesh renderer...")
        self.mesh_renderer = NVDiffRenderer(use_opengl=False)
        
        self.pipe = PipelineConfig()
        # recording settings
        self.keyframes = []  # list of state dicts of keyframes
        self.all_frames = {}  # state dicts of all frames {key: [num_frames, ...]}
        self.num_record_timeline = 0
        self.playing = False

        print("Initializing 3D Gaussians...")
        self.init_gaussians()
        
        # FLAME parameters
        print("Initializing FLAME parameters...")
        self.reset_flame_param()
        
        print("Initializing GUI...")

        if self.gaussians.binding != None:
            self.num_timesteps = self.gaussians.num_timesteps

            self.gaussians.select_mesh_by_timestep(self.timestep)

    
    def init_gaussians(self):
        # load gaussians
        # self.gaussians = GaussianModel(self.cfg.sh_degree)
        self.gaussians = FlameGaussianModel(3)

        # selected_fid = self.gaussians.flame_model.mask.get_fid_by_region(['left_half'])
        # selected_fid = self.gaussians.flame_model.mask.get_fid_by_region(['right_half'])
        # unselected_fid = self.gaussians.flame_model.mask.get_fid_except_fids(selected_fid)
        unselected_fid = []
        
        self.gaussians.load_ply('GA_output/point_cloud/iteration_600000/point_cloud.ply', has_target=False, motion_path='GA_output/point_cloud/iteration_600000/flame_param.npz', disable_fid=unselected_fid)
  


    def reset_flame_param(self):
        self.flame_param = {
            'expr': torch.zeros(1, self.gaussians.n_expr),
            'rotation': torch.zeros(1, 3),
            'neck': torch.zeros(1, 3),
            'jaw': torch.zeros(1, 3),
            'eyes': torch.zeros(1, 6),
            'translation': torch.zeros(1, 3),
        }

    def prepare_camera(self):
        @dataclass
        class Cam:
            FoVx = float(np.radians(self.cam.fovx))
            FoVy = float(np.radians(self.cam.fovy))
            image_height = self.cam.image_height
            image_width = self.cam.image_width
            world_view_transform = torch.tensor(self.cam.world_view_transform).float().cuda().T  # the transpose is required by gaussian splatting rasterizer
            full_proj_transform = torch.tensor(self.cam.full_proj_transform).float().cuda().T  # the transpose is required by gaussian splatting rasterizer
            camera_center = torch.tensor(self.cam.pose[:3, 3]).cuda()
        return Cam

    @torch.no_grad()
    def run(self):
        print("Running GaussianSplattingViewer...")

        faces = self.gaussians.faces.clone()
        # faces = faces[selected_fid]
        frames = np.load('verts.npy')
        cam = self.prepare_camera()
        start= 0
        for frame in frames:
                vertss  = torch.from_numpy(frame.reshape(5023, 3)).cuda()
                vertss = self.gaussians.flame_model.add_teeth_vert(vertss)
                vertss = vertss.view(1, -1, 3)

                self.gaussians.update_mesh_properties(vertss, vertss)
                # self.gaussians.update_mesh_by_param_dict(self.flame_param)

                if self.show_spatting:
                    # rgb
                    rgb_splatting = render(cam, self.gaussians, self.pipe, torch.tensor((1., 1., 1.)).cuda(), scaling_modifier=self.scaling_modifier)["render"].permute(1, 2, 0).contiguous()

                    # opacity
                    # override_color = torch.ones_like(self.gaussians._xyz).cuda()
                    # background_color = torch.tensor(self.cfg.background_color).cuda() * 0
                    # rgb_splatting = render(cam, self.gaussians, self.cfg.pipeline, background_color, scaling_modifier=self.scaling_modifier, override_color=override_color)["render"].permute(1, 2, 0).contiguous()

                if self.gaussians.binding != None and self.show_mesh:
                    out_dict = self.mesh_renderer.render_from_camera(self.gaussians.verts, faces, cam)

                    rgba_mesh = out_dict['rgba'].squeeze(0)  # (H, W, C)
                    rgb_mesh = rgba_mesh[:, :, :3]
                    alpha_mesh = rgba_mesh[:, :, 3:]

                    mesh_opacity = self.mesh_color[3:].cuda()
                    mesh_color = self.mesh_color[:3].cuda()
                    rgb_mesh = rgb_mesh * (alpha_mesh * mesh_color + (1 - alpha_mesh))

                if self.show_spatting and self.show_mesh:
                    rgb = rgb_mesh * alpha_mesh * mesh_opacity  + rgb_splatting * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
                elif self.show_spatting and not self.show_mesh:
                    rgb = rgb_splatting
                elif not self.show_spatting and self.show_mesh:
                    rgb = rgb_mesh
                else:
                    rgb = torch.ones([self.H, self.W, 3])

                # image_np = image.permute(1, 2, 0).detach().cpu().numpy()

                image=  rgb.cpu().numpy()
                image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))

                # Plot the image
                # plt.imshow(portrait, cmap='gray')
                plt.axis('off')  # Turn off axis
                plt.title('Random Image')

                # Save the image to a file
                plt.imsave(f'output/frame_{start:03d}.png', image_normalized)
                start += 1





if __name__ == "__main__":
    gui = GaussianSplattingViewer()
    gui.run()
