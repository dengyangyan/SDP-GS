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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getWorld2View_project

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, point_feature, seg_map, feature_dict, gt_alpha_mask, depth_mono,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", bounds=None, intr=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.bounds = bounds

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.depth_mono = None
        if depth_mono is not None:
            self.depth_mono = depth_mono.to(self.data_device)
        
        


        if image is not None:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]
        self.point_feature = point_feature.to(self.data_device) if point_feature is not None else None
        self.seg_map = seg_map.to(self.data_device) if seg_map is not None else None
        self.feature_dict = feature_dict.to(self.data_device) if feature_dict is not None else None


        # self.image_width = self.original_image.shape[2]
        # self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        # else:
        #     self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        R_l = R.copy()
        # R_l[0, 1] = -R_l[0, 1]
        # R_l[1, 0] = -R_l[1, 0]
        # R_l[1, 2] = -R_l[1, 2]
        # R_l[2, 1] = -R_l[2, 1]
        # R_l[:2, 0] = -R_l[:2, 0]
        # R_l[0, :2] = -R_l[0, :2]
        self.world_view_transform_project = torch.tensor(getWorld2View_project(R_l, T, trans, scale)).cuda()

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.intrinsics = torch.tensor(intr).to(self.data_device) if intr is not None else None
        # 后续处理统一添加
        self.feature_map_all = None


class PseudoCamera(nn.Module):
    def __init__(self, R, T, FoVx, FoVy, width, height, intr=None, trans=np.array([0.0, 0.0, 0.0]), scale=1.0 ):
        super(PseudoCamera, self).__init__()

        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = width
        self.image_height = height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform_project = torch.tensor(getWorld2View_project(R, T, trans, scale)).transpose(0, 1).cuda()

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.intrinsics = torch.tensor(intr).cuda() if intr is not None else None




class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
