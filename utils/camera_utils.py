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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, NPtoTorch
from utils.graphics_utils import fov2focal
import torch.nn.functional as F
import PIL.Image as Image
import cv2

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    # convert intrinsics to new resolution
    cam_info.intr[0, :] /= (orig_w/resolution[0])
    cam_info.intr[1, :] /= (orig_h/resolution[1])



    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    if cam_info.point_feature is not None: # if we have a point feature, we have a depth map
        resized_depth_mono = NPtoTorch(cam_info.depth_mono, resolution)

    else:
        if isinstance (cam_info.depth_mono, np.ndarray):
            resized_depth_mono = Image.fromarray(cam_info.depth_mono)
        else:
            resized_depth_mono = cam_info.depth_mono
        resized_depth_mono = PILtoTorch(resized_depth_mono, resolution)




    # order=1 for bilinear interpolation
    if cam_info.point_feature is not None:
        # resize point feature
        point_feature = F.interpolate(cam_info.point_feature.permute(2, 0, 1).unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False).squeeze(0)
        # point_feature = cam_info.point_feature
        # seg_map = cam_info.seg_map.squeeze(0)
        seg_map = F.interpolate(cam_info.seg_map.unsqueeze(0).float(), size=(resolution[1], resolution[0]), mode='nearest').squeeze().long()
        feature_dict = cam_info.feature_dict

    else:
        point_feature = None
        seg_map = None
        feature_dict = None




    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, point_feature=point_feature, seg_map=seg_map, feature_dict=feature_dict,
                  gt_alpha_mask=loaded_mask, depth_mono=resized_depth_mono, 
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, bounds=cam_info.bounds, intr = cam_info.intr)

def loadRenderCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 6400:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 6.4K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 6400
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    point_feature = None
    seg_map = None
    feature_dict = None



    cam = Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=None, point_feature=point_feature, seg_map=seg_map, feature_dict=feature_dict,
                  gt_alpha_mask=None, depth_mono=None, 
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, bounds=cam_info.bounds, intr = cam_info.intr)
    cam.image_width, cam.image_height = resolution
    return cam



def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def renderCameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadRenderCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
