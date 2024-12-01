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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text, read_points3D_binary_sparse
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.loss_utils import normalize_seg
from scene.gaussian_model import BasicPointCloud
import torch, cv2
import glob
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import re
def read_pfm(path):
    """Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale


# copy from monosdf
def compute_scale_and_shift(prediction, target, mask):
    mask_prediction = (prediction > 0.1) & (prediction < 30.0)
    mask_target = (target > 0.1) & (target < 30.0)
    mask = mask & mask_prediction & mask_target

    # plt.scatter(prediction[mask], target[mask])
    # plt.show()
    
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = np.sum(mask * prediction * prediction, (0, 1))/mask.sum()
    a_01 = np.sum(mask * prediction, (0, 1))/mask.sum()
    a_11 = np.sum(mask, (0, 1))/mask.sum()

    # right hand side: b = [b_0, b_1]
    b_0 = np.sum(mask * prediction * target, (0, 1))/mask.sum()
    b_1 = np.sum(mask * target, (0, 1))/mask.sum()

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = np.zeros_like(b_0)
    x_1 = np.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    x_0 = (a_11 * b_0 - a_01 * b_1) / det
    x_1 = (-a_01 * b_0 + a_00 * b_1) / det

    prediction_new = prediction * x_0 + x_1
    # plt.scatter(prediction[mask], target[mask])
    # plt.show()
    # plt.figure()
    # plt.scatter(prediction_new[mask], target[mask])
    # plt.show()
    # valid = det.nonzero()

    # x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    # x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return prediction_new

def plot_mask(img, masks, colors=None, alpha=0.5) -> np.ndarray:
   """Visualize segmentation mask.

    Parameters
    ----------
    img: numpy.ndarray
        Image with shape `(H, W, 3)`.
    masks: numpy.ndarray
        Binary images with shape `(N, H, W)`.
    colors: numpy.ndarray
        color for mask, shape `(N, 3)`.
        if None, generate random color for mask
    alpha: float, optional, default 0.5
        Transparency of plotted mask

    Returns
    -------
    numpy.ndarray
        The image plotted with segmentation masks, shape `(H, W, 3)`

    """
   if colors is None:
      colors = np.random.random((masks.shape[0], 3)) * 255
   else:
      if colors.shape[0] < masks.shape[0]:
         raise RuntimeError(
            f"colors count: {colors.shape[0]} is less than masks count: {masks.shape[0]}"
         )
   for mask, color in zip(masks, colors):
      mask = np.stack([mask, mask, mask], -1)
      img = np.where(mask, img * (1 - alpha) + color * alpha, img)

   return img.astype(np.uint8)
def generalID(x,y,column_num,row_num,x_min,x_max,y_min,y_max):
    # 若在范围外的点，返回-1
    if x < x_min or x > x_max or y < y_min or y > y_max:
        return -1
    # 把范围根据列数等分切割
    column = (x_max - x_min)/column_num
    # 把范围根据行数等分切割
    row = (y_max - y_min)/row_num
    # 得到二维矩阵坐标索引，并转换为一维ID，即： 列坐标区域（向下取整）+ 1 + 行坐标区域 * 列数
    return int((x-x_min)/column)+ 1 + int((y-y_min)/row) * column_num

def compare(depth_ALL, depth_ALL_sparse, depth_mono, seg_maps=None, image=None):
    """
    比较depth_ALL和depth_mono的差异
    """
    depth_mono = 1 - np.array(depth_mono)/255.0
    depth_ALL = cv2.resize(np.array(depth_ALL, dtype=np.float32), (depth_mono.shape[1], depth_mono.shape[0]))

    depth_ALL_mask = np.stack([depth_ALL, depth_ALL, depth_ALL], -1)
    seg_maps = cv2.resize(np.array(seg_maps.squeeze().cpu().numpy(), dtype=np.float32), (depth_mono.shape[1], depth_mono.shape[0]), interpolation=cv2.INTER_NEAREST)
    image = np.array(image)/255.0

    depth_ALL_ = depth_ALL[(depth_ALL > 0)&(depth_ALL < 0.99*depth_ALL.max())].reshape(-1, 1)
    depth_mono_ALL = depth_mono[(depth_ALL > 0)&(depth_ALL < 0.99*depth_ALL.max())].reshape(-1, 1)
    # upscale to the same size with depth_ALL
    depth_ALL_sparse_ = depth_ALL_sparse[depth_ALL_sparse > 0]

    depth_mono_sparse = depth_mono[depth_ALL_sparse > 0]

    # 展示depth_ALL和depth_mono的概率分布

    # 将两个维度的数据进行量化，然后生成曲线密度图
    d = np.vstack([depth_ALL_, depth_mono_ALL])
    xmin, xmax = depth_ALL_.min(), depth_ALL_.max()
    ymin, ymax = depth_mono_ALL.min(), depth_mono_ALL.max()
    rows, cols = 100, 100
    x, y = np.linspace(xmin, xmax, cols), np.linspace(ymin, ymax, rows)
    x, y = np.meshgrid(x, y)
    z = np.zeros_like(x)
    for i in range(rows):
        for j in range(cols):
            z[i, j] = np.sum((depth_ALL_ >= x[i, j]) & (depth_ALL_ < x[i, j] + (xmax-xmin)/cols) & (depth_mono_ALL >= y[i, j]) & (depth_mono_ALL < y[i, j] + (ymax-ymin)/rows))
    # 拟合线性方程

    from sklearn.linear_model import (LinearRegression, HuberRegressor, RANSACRegressor, TheilSenRegressor)
    import pandas as pd
    lr = LinearRegression().fit(depth_ALL_, depth_mono_ALL)
    coef_list = [["lr_coef", float(lr.coef_[0])]]
    plotline_X = np.arange(depth_ALL_.min(), depth_ALL_.max(), 0.1).reshape(-1, 1)
    fit_df = pd.DataFrame( index = plotline_X[:, 0], data={"linear_regression": lr.predict(plotline_X).flatten()})


    ransac = RANSACRegressor(random_state=42).fit(depth_ALL_, depth_mono_ALL)
    fit_df["ransac_regression"] = ransac.predict(plotline_X)
    ransac_coef = ransac.estimator_.coef_
    coef_list.append(["ransac_regression", ransac.estimator_.coef_[0]])
    inlier_mask = ransac.inlier_mask_
    outlier_mask = ~inlier_mask
    print(f"Total outliers: {sum(outlier_mask)/len(depth_ALL_)}")


    theilsen = TheilSenRegressor(random_state=42).fit(depth_ALL_, depth_mono_ALL)
    fit_df["theilsen_regression"] = theilsen.predict(plotline_X)
    coef_list.append(["theilsen_regression", theilsen.coef_[0]])
    # print(f"Outliers you added yourself: {sum(outlier_mask[:N_OUTLIERS])} / {N_OUTLIERS}")

    huber = HuberRegressor().fit(depth_ALL_, depth_mono_ALL)
    fit_df["huber_regression"] = huber.predict(plotline_X)
    coef_list.append(["huber_regression", huber.coef_[0]])

    fix, ax = plt.subplots()
    plt.contourf(x, y, z, alpha=0.75, cmap='binary')
    fit_df.plot(ax=ax, colormap='coolwarm')

    plt.colorbar()
    plt.xlabel('dense_nearlyGT')
    plt.ylabel('monodepth')
    plt.savefig("depth_map_compare.jpg")

    """
    比较同一个segmentation mask下的depth map和depth_mono的差异
    """
    total_image_show = np.zeros_like(image)
    image = np.array(image)/255
    alpha = 0.5
    for seg_map_idx in np.unique(seg_maps):
        mask = seg_maps == seg_map_idx
        color = np.random.rand(3)
        mask_image = np.stack([mask, mask, mask], -1)
        img = np.where(mask_image, image * (1 - alpha) + color * alpha, image)
        # image_show = plot_mask(image, mask, colors=[color])
        plt.subplot(1, 2, 1)
        plt.cla()
        plt.imshow(img)
        depth_map = depth_ALL[mask&(depth_ALL > 0)]
        depth_mono_seg = depth_mono[mask &(depth_ALL > 0)]
        for i in range(rows):
            for j in range(cols):
                z[i, j] = np.sum((depth_map >= x[i, j]) & (depth_map < x[i, j] + (xmax-xmin)/cols) & (depth_mono_seg >= y[i, j]) & (depth_mono_seg < y[i, j] + (ymax-ymin)/rows))

        plt.subplot(1, 2, 2)
        plt.cla()
        plt.contourf(x, y, z, alpha=0.75, cmap='binary')

        # # 设置横轴和纵轴的范围
        # plt.xlim([0, depth_ALL_.max()])
        # plt.ylim([0, 1])
        plt.savefig(f"depth_map_compare_{seg_map_idx}.jpg")
        # plt.scatter(depth_map, depth_mono_seg, c=color)
        # plt.show()

    return


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    point_feature: np.array
    seg_map: np.array
    feature_dict: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    bounds: np.array
    intr: np.array
    depth_mono: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def get_language_feature_threemasks_torch(language_feature_dir, image_name, image_height, image_width, device):
    language_feature_name = os.path.join(language_feature_dir, image_name)
    seg_map = torch.from_numpy(np.load(language_feature_name + '_s.npy')).to(device)# .unsqueeze(0)
    feature_dict_ori = torch.from_numpy(np.load(language_feature_name + '_fdim3.npy')).to(device)
    


    # """
    # 分割时的w和h
    # """
    # seg_h = 378
    # seg_w = 504

    # y, x = torch.meshgrid(torch.arange(0, seg_h), torch.arange(0, seg_w))
    # x = x.reshape(-1, 1).cuda()
    # y = y.reshape(-1, 1).cuda()
    # seg = seg_map[:, y, x].squeeze(-1).long()
    # # output = F.interpolate(input, size=(h, w), mode='bilinear', align_corners=False)
    # seg = F.interpolate(seg.unsqueeze(0).unsqueeze(0).float(), size=(image_height, image_width), mode='nearest').squeeze(0).squeeze(0).long()


    seg_map = F.interpolate(seg_map.unsqueeze(0).float(), size=(image_height, image_width), mode='nearest').squeeze(0).long()
        # # 存储point_feature
    cv2.imwrite(language_feature_name + '_f.png', ((seg_map[0]+1)/2).cpu().numpy())
    # # 去面积过小的区域及其featurea , 
    # feature_dict = []
    # for idx, i in enumerate(seg_map.unique()):
    #     if (seg_map==i).sum() < (image_height * image_width * 5e-3): # blender 5e-3 # LLFF DTU1e-8
    #         seg_map[seg_map==i] = -1
    #     else:
    #         feature_dict.append(feature_dict_ori[idx])
    # feature_dict = torch.stack(feature_dict, dim=0)

    # if seg_map.min() == -1:
    #     # 膨胀segmap=-1的区域的id众数
    #     seg_map_below = (seg_map.clone().cpu().numpy()==-1).astype(np.uint8)
    #     # cv2.imwrite("seg_map_below.jpg", seg_map_below[0]*255)
    #     seg_map_below_ = cv2.dilate(seg_map_below[0], np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    #     seg_map_below_[seg_map_below[0].astype(bool)] = False
    #     # cv2.imwrite("seg_map_below_.jpg", seg_map_below_*255)
        
    #     seg_map_below_ = torch.from_numpy(seg_map_below_).to(device).unsqueeze(0)
    #     # if seg_map[seg_map_below_].mode().values==-1:
    #     #     print("seg_map_below_ mode is -1")
    #     # 计算seg_map[seg_map_below]的众数
    #     seg_map[seg_map==-1] = seg_map[seg_map_below_].mode().values
            


    mask = (seg_map != -1).reshape(1, image_height, image_width)


    # 把seg转换为从小到大的连续的id
    max_id_in_all = torch.unique(seg_map)
    if max_id_in_all[0] == -1:
        max_id_in_all = max_id_in_all[1:]
    seg_new_order = torch.zeros_like(seg_map.view(-1), dtype=torch.int32).to(device)
    for idx, i in enumerate(max_id_in_all):
        seg_new_order[seg_map.view(-1)==i] = idx

    # test
    seg_map_new = seg_new_order.view(image_height, image_width)
    cv2.imwrite(language_feature_name + '_f.png', (seg_map_new.float()/seg_map_new.max()*255).cpu().numpy())

    point_feature1 = feature_dict_ori[seg_new_order]

    point_feature = point_feature1.view(image_height, image_width, -1)

    if seg_map.min() == -1:
        print("seg_map min is -1")
    
    # # 存储point_feature
    cv2.imwrite(language_feature_name + '_f.png', ((point_feature+1)/2).cpu().numpy()*255)


    return point_feature, mask, feature_dict_ori, seg_map # , text_map

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, features_folder, path, rgb_mapping, train_cam_names):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        bounds = np.load(os.path.join(path, 'poses_bounds.npy'))[idx, -2:]

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            if intr.model=="SIMPLE_RADIAL":
                k = intr.params[3]
            # intrinsics计算
            intrinsics = np.array([[intr.params[0], 0, intr.params[1], 0], [0, intr.params[0], intr.params[2], 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            intrinsics = np.array([[intr.params[0], 0, intr.params[2], 0], [0, intr.params[1], intr.params[3], 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        # rgb_path = rgb_mapping[idx]
        image = Image.open(image_path)

        
        if image_name in train_cam_names:
        # if os.path.exists(features_folder):
            point_feature, mask, feature_dict, seg_map= get_language_feature_threemasks_torch(features_folder, image_name, image.height, image.width, device=torch.device('cuda'))
            text_map = None
        else: 
            point_feature, mask, feature_dict, seg_map, text_map = None, None, None, None, None # mask为seg_map的not -1
        
        # # for all views
        # point_feature, mask, feature_dict, seg_map, text_map = None, None, None, None, None # mask为seg_map的not -1

        if image_name in train_cam_names:
            depth_mono_path = os.path.join('/'.join(images_folder.split("/")[:-1]), 'depth_adjust_maps_stereo', 'depth_' + os.path.basename(extr.name).split(".")[0] + '.npy')
            # # # print(image_name)
            depth_mono = np.load(depth_mono_path)
            depth_mono = cv2.resize(depth_mono, (width, height), interpolation=cv2.INTER_NEAREST)

            # # for baseline
            # depth_mono_path = os.path.join('/'.join(images_folder.split("/")[:-1]), 'depth_maps', 'depth_' + os.path.basename(extr.name).split(".")[0] + '.pfm')
            # depth_mono = read_pfm(depth_mono_path)[0]    
            # depth_mono = cv2.resize(depth_mono, (width, height), interpolation=cv2.INTER_NEAREST)

            # cv2.imwrite("f_depth_mono.jpg", (depth_mono-depth_mono.min())/(depth_mono.max()-depth_mono.min())*255)

        else:
            depth_mono_path = os.path.join('/'.join(images_folder.split("/")[:-1]), 'depth_maps', 'depth_' + os.path.basename(extr.name).split(".")[0] + '.png')
            depth_mono = cv2.imread(depth_mono_path, cv2.IMREAD_GRAYSCALE)
            depth_mono = cv2.resize(depth_mono, (width, height), interpolation=cv2.INTER_NEAREST)


        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, point_feature=point_feature, seg_map=seg_map, feature_dict=feature_dict,
                              image_path=image_path, image_name=image_name, width=width, height=height, bounds=bounds, intr=intrinsics, depth_mono=depth_mono)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos
def topk_(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data,axis=axis)
        topk_data_sort = topk_data[topk_index_sort,row_index]
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return topk_data_sort

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = None
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, features, eval, n_views=3, llffhold=8, dataset = "LLFF"): # 8
    
    ply_path = os.path.join(path, str(n_views) + "_views/dense/fused.ply")
    # ply_path = os.path.join(path, "mvsnet_fern_downsample.ply")

    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    rand_pcd = False

    if not os.path.exists(ply_path):
        print("No fused point cloud found, generating random point cloud.")
        rand_pcd = True

    if rand_pcd:
        print('Init random point cloud.')
        ply_path = os.path.join(path, "sparse/0/points3D_random.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")

        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        print(xyz.max(0), xyz.min(0))

        if dataset == "DTU":
            pcd_shape = (topk_(xyz, 100, 0)[-1] + topk_(-xyz, 100, 0)[-1])
            num_pts = 10_00
            xyz = np.random.random((num_pts, 3)) * pcd_shape * 1.3 - topk_(-xyz, 100, 0)[-1] # - 0.15 * pcd_shape
        
        else:
            pcd_shape = (topk_(xyz, 1, 0)[-1] + topk_(-xyz, 1, 0)[-1])
            num_pts = int(pcd_shape.max() * 50)
            xyz = np.random.random((num_pts, 3)) * pcd_shape * 1.3 - topk_(-xyz, 20, 0)[-1]

        print(pcd_shape)
        print(f"Generating random point cloud ({num_pts})...")

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        pcd = fetchPly(ply_path)


    reading_dir = "images" if images == None else images
    reading_dim3_dir = 'DINO_features_GGrouping_dim3' if features == None else features
    rgb_mapping = [f for f in sorted(glob.glob(os.path.join(path, reading_dir, '*')))
                   if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

    if dataset == "DTU":
        cam_extrinsics = {cam_extrinsics[k].name: cam_extrinsics[k] for k in cam_extrinsics}
    else:
        cam_extrinsics = sorted(cam_extrinsics.items(), key=lambda x: int(x[1].name.split('.')[0][4:]))
        cam_extrinsics = {k[1].name: k[1] for k in cam_extrinsics}    


    if n_views > 0:
        if dataset == "DTU":
            train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
            exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
            test_idx = [i for i in np.arange(49) if i not in train_idx + exclude_idx]
            cam_names = [c for idx, c in enumerate([cam_extrinsics[k].name.split('.')[0] for k in cam_extrinsics])]
            sort_names = sorted(cam_names) 
            if n_views > 0:
                train_idx = train_idx[:n_views]
            train_cam_names = [c for idx, c in enumerate(sort_names) if idx in train_idx]
        else:
            if eval:
                train_cam_names_tmp = [c for idx, c in enumerate([cam_extrinsics[k].name.split('.')[0] for k in cam_extrinsics]) if idx % llffhold != 0]
            else:
                train_cam_names_tmp = [c for idx, c in enumerate([cam_extrinsics[k].name.split('.')[0] for k in cam_extrinsics])]

            idx_sub = np.linspace(0, len(train_cam_names_tmp)-1, n_views)
            idx_sub = [round(i) for i in idx_sub]
            train_cam_names = [c for idx, c in enumerate(train_cam_names_tmp) if idx in idx_sub]
            
            # # for all views
            # train_cam_names_tmp = [c for idx, c in enumerate([cam_extrinsics[k].name.split('.')[0] for k in cam_extrinsics])]
            # train_cam_names = [c for idx, c in enumerate(train_cam_names_tmp)]
            # print(len(train_cam_names))
            assert len(train_cam_names) == n_views


    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics,
                                        cam_intrinsics=cam_intrinsics,
                                        images_folder=os.path.join(path, reading_dir),
                                        features_folder=os.path.join(path, reading_dim3_dir),
                                        path=path, rgb_mapping=rgb_mapping, train_cam_names=train_cam_names)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        print("Dataset Type: ", dataset)
        if dataset == "DTU":
            train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
            exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
            test_idx = [i for i in np.arange(49) if i not in train_idx + exclude_idx]
            if n_views > 0:
                train_idx = train_idx[:n_views]
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_idx]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_idx]
        else:
            print("LLFF Holdout")
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
            if n_views > 0:
                idx_sub = np.linspace(0, len(train_cam_infos)-1, n_views)
                idx_sub = [round(i) for i in idx_sub]
                train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_sub]
                assert len(train_cam_infos) == n_views
        # else:
        #     raise NotImplementedError
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)


    # ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    path_images = os.path.join(path, "sparse/0/images.bin")
    # if not os.path.exists(ply_path):
    #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    #     try:
    #         xyz, rgb, _ = read_points3D_binary(bin_path)
    #     except:
    #         xyz, rgb, _ = read_points3D_text(txt_path)
    #     storePly(ply_path, xyz, rgb)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
def readCamerasFromTransforms(path, transformsfile, white_background, features_folder, train_cam_names, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            feature_path = os.path.join(features_folder, image_name+'_s.npy')
            # if image_name in train_cam_names:
            if os.path.exists(feature_path):
                point_feature, mask, feature_dict, seg_map= get_language_feature_threemasks_torch(features_folder, image_name, image.height, image.width, device=torch.device('cuda'))

            else: 
                point_feature, mask, feature_dict, seg_map = None, None, None, None # mask为seg_map的not -1



            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx
            fx = fov2focal(fovx, image.size[0])
            fy = fov2focal(fovy, image.size[1])
            cx = image.size[0] / 2
            cy = image.size[1] / 2
            intrinsics = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
            width, height = image.size[0], image.size[1]
            
            # # monocular depth
            # ## (fix path)
            # depth_mono_path = os.path.join('/'.join(image_path.split("/")[:-1]), 'depth_maps', 'depth_' + image_name + '.png')
            # depth_mono = cv2.imread(depth_mono_path, cv2.IMREAD_GRAYSCALE)
            # depth_mono = Image.fromarray(depth_mono)
            

            if (image_name in train_cam_names) and "train" in image_path:
                # depth_mono_path = os.path.join('/'.join(image_path.split("/")[:-1]), 'depth_adjust_maps_stereo', 'depth_' + image_name + '.npy')
                # print(image_name)
                # depth_mono = np.load(depth_mono_path)
                # depth_mono = cv2.resize(depth_mono, (width, height), interpolation=cv2.INTER_NEAREST)

                # for baseline
                depth_mono_path = os.path.join('/'.join(image_path.split("/")[:-1]), 'depth_maps', 'depth_' + image_name  + '.pfm')
                depth_mono = read_pfm(depth_mono_path)[0]    
                depth_mono = depth_mono.max() - depth_mono
                depth_mono = cv2.resize(depth_mono, (width, height), interpolation=cv2.INTER_CUBIC)

                cv2.imwrite("f_depth_mono.jpg", (depth_mono-depth_mono.min())/(depth_mono.max()-depth_mono.min())*255)

            else:
                depth_mono_path = os.path.join('/'.join(image_path.split("/")[:-1]), 'depth_maps', 'depth_' + image_name + '.png')
                depth_mono = cv2.imread(depth_mono_path, cv2.IMREAD_GRAYSCALE)
                depth_mono = cv2.resize(depth_mono, (width, height), interpolation=cv2.INTER_NEAREST)




            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, point_feature=point_feature, seg_map=seg_map, feature_dict=feature_dict,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], bounds=None, intr=intrinsics, depth_mono=depth_mono))
            
    return cam_infos


def readCamerasFromTransforms_mipnerf(path, transformsfile, white_background, features_folder, cam_names, mode='train'):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fx = contents["fl_x"]
        fy = contents["fl_y"]
        w = contents["w"]
        h = contents["h"]
        fovx = focal2fov(fx, w)
        cx = contents["cx"]
        cy = contents["cy"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"])
            image_name = cam_name.split('/')[-1]
            if image_name in cam_names:

                # NeRF 'transform_matrix' is a camera-to-world transform
                c2w = np.array(frame["transform_matrix"])
                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                c2w[:3, 1:3] *= -1

                # get the world-to-camera transform and set R, T
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]

                image_path = os.path.join(path, cam_name)
                image_name = Path(cam_name).stem
                image = Image.open(image_path)

                im_data = np.array(image.convert("RGBA"))
                point_feature, mask, feature_dict, seg_map, text_map = None, None, None, None, None # mask为seg_map的not -1

                if features_folder != None:
                    if mode == 'train':
                        point_feature, mask, feature_dict, seg_map= get_language_feature_threemasks_torch(features_folder, image_name, image.size[1], image.size[0])
                        text_map = None
        


                bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
                w = image.size[0]   
                h = image.size[1]
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy 
                FovX = fovx
                fx = fov2focal(fovx, image.size[0])
                fy = fov2focal(fovy, image.size[1])
                # fx = 2*np.tan(fovx/2)/w
                # fy = 2*np.tan(fovy/2)/h
                
                # fov_x = np.rad2deg(2 * np.arctan2(w, 2 * fx))
                # fov_y = np.rad2deg(2 * np.arctan2(h, 2 * fy))
                # cx = image.size[0] / 2
                # cy = image.size[1] / 2
                intrinsics = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
                bounds = None

                cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, point_feature=point_feature, seg_map=seg_map, feature_dict=feature_dict,
                                image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], bounds=bounds, intr=intrinsics))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, features_name, llffhold=8, n_views=8, extension=".png"):

    train_cam_names = []
    with open(os.path.join(path, 'transforms_train.json')) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = frame["file_path"].split('/')[-1]
            train_cam_names.append(cam_name)
            
    if eval:
        if n_views > 0:
            train_cam_names = [c for idx, c in enumerate(train_cam_names) if idx in [2, 16, 26, 55, 73, 76, 86, 93]]



    reading_dim3_dir = 'language_features_GGrouping_dim3' if features_name == None else features_name
    features_dir = os.path.join(path, reading_dim3_dir)
    
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, features_dir, train_cam_names, extension)
    
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, features_dir, train_cam_names, extension)
    
    if eval:
        if n_views > 0:
            train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in [2, 16, 26, 55, 73, 76, 86, 93]]
        eval_cam_infos = [c for idx, c in enumerate(test_cam_infos) if idx % llffhold == 0]
        test_cam_infos = test_cam_infos
    else:
        test_cam_infos = []
        eval_cam_infos = []


    # print('train', [info.image_path for info in train_cam_infos])
    # print('eval', [info.image_path for info in eval_cam_infos])

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # ply_path = 
    ply_path = os.path.join(path, f"{n_views}_views/dense/fused.ply")
    try :
        pcd = fetchPly(ply_path)
        if pcd.points.shape[0] < 1000:
            print("Point cloud is too small, generating random point cloud.")
            print(pcd.points.shape)
            pcd = None
    except:
        pcd = None
    
    


    if (pcd == None):
        ply_path = os.path.join(path, "points3d.ply")

        if not os.path.exists(ply_path):
            # Since this data set has no colmap data, we start with random points
            num_pts = 100_00# 0
            print(f"Generating random point cloud ({num_pts})...")
            
            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

            storePly(ply_path, xyz, SH2RGB(shs) * 255)
            
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
    

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=eval_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readmipnerf360Info(path, white_background, eval, features_name, llffhold=8, n_views=3, extension=".png"):

    with open(os.path.join(path, f'train_test_split_{n_views}.json')) as json_file:
        contents = json.load(json_file)
        test_split = contents["test_ids"]
        train_split = contents["train_ids"]


    train_cam_names = []
    test_cam_names = []
    with open(os.path.join(path, 'transforms.json')) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = frame["file_path"].split('/')[-1]
            if idx in train_split:
                train_cam_names.append(cam_name)
            elif idx in test_split:
                test_cam_names.append(cam_name)


    reading_dim3_dir = 'language_features_GGrouping_dim3' if features_name == None else features_name
    features_dir = os.path.join(path, reading_dim3_dir)
    
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms_mipnerf(path, "transforms.json", white_background, features_dir, train_cam_names, mode='train')
    
    print("Reading Test Transforms")
    eval_cam_infos = readCamerasFromTransforms_mipnerf(path, "transforms.json", white_background, None, test_cam_names, mode='test')
    


    print('train', [info.image_path for info in train_cam_infos])
    print('eval', [info.image_path for info in eval_cam_infos])

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_00
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=eval_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def generateLLFFCameras(poses):
    cam_infos = []
    Rs, tvecs, height, width, focal_length_x = pose_utils.convert_poses(poses) 
    # print(Rs, tvecs, height, width, focal_length_x)
    for idx, _ in enumerate(Rs):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(Rs)))
        sys.stdout.flush()

        uid = idx
        R = np.transpose(Rs[idx])
        T = tvecs[idx]

        FovY = focal2fov(focal_length_x, height)
        FovX = focal2fov(focal_length_x, width)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=None, point_feature=None, seg_map=None, feature_dict=None,
                              image_path=None, image_name=None, width=width, height=height, bounds=None, intr=None, depth_mono=None)


        # cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=None, depth_mono=None, 
        #                       image_path=None, image_name=None, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

from utils import pose_utils


def CreateLLFFSpiral(basedir):

    # Load poses and bounds.
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses_o = poses_arr[:, :-2].reshape([-1, 3, 5])
    bounds = poses_arr[:, -2:]
    
    # Pull out focal length before processing poses.
    # Correct rotation matrix ordering (and drop 5th column of poses).
    fix_rotation = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
                            dtype=np.float32)
    inv_rotation = np.linalg.inv(fix_rotation)
    poses = poses_o[:, :3, :4] @ fix_rotation

    # Rescale according to a default bd factor.
    # scale = 1. / (bounds.min() * .75)
    # poses[:, :3, 3] *= scale
    # bounds *= scale

    # Recenter poses.
    render_poses, _ = pose_utils.recenter_poses(poses)

    # Separate out 360 versus forward facing scenes.
    render_poses = pose_utils.render_generate_spiral_path(
          render_poses, bounds, n_frames=180)
    render_poses = pose_utils.backcenter_poses(render_poses, poses)
    render_poses = render_poses @ inv_rotation
    render_poses = np.concatenate([render_poses, np.tile(poses_o[:1, :3, 4:], (render_poses.shape[0], 1, 1))], -1)

    render_cam_infos = generateLLFFCameras(render_poses.transpose([1,2,0]))

    nerf_normalization = getNerfppNorm(render_cam_infos)

    scene_info = SceneInfo(point_cloud=None,
                           train_cameras=None,
                           test_cameras=render_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=None)
    return scene_info



sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "mipnerf360" : readmipnerf360Info,
    "Blender" : readNerfSyntheticInfo,
    "Spiral" : CreateLLFFSpiral,    
}