import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import sys
import re
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image
import matplotlib.pyplot as plt
from compare_llff import read_pfm, read_stereo_sparse_points, compute_scale_and_shift
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse. May be different from the original implementation')
parser.add_argument('--model', default='mvsnet', help='select model')

parser.add_argument('--dataset', default='dtu_yao_eval', help='select dataset')
parser.add_argument('--testpath', help='testing data path')
parser.add_argument('--testlist', help='testing scan list')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the depth interval scale')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--outdir', default='/home/dengyangyan/code/data/nerf_llff_data/flower', help='output dir')
parser.add_argument('--display', action='store_true', help='display depth images and masks')

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def write_pfm(file, image, scale=1):
    file = open(file, 'wb')
    color = None
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode() if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image_string = image.tostring()
    file.write(image_string)
    file.close()


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4
    return intrinsics, extrinsics


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            data.append((ref_view, src_views))
    return data


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < 5, relative_depth_diff < 0.2)
    # mask = np.logical_and(dist < 0.1, relative_depth_diff < 0.001)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src
# 点云降采样
def downsample(points, colors, voxel_size):
    voxel_grid = {}
    for i, point in enumerate(points):
        voxel_x = int(point[0] / voxel_size)
        voxel_y = int(point[1] / voxel_size)
        voxel_z = int(point[2] / voxel_size)
        voxel = (voxel_x, voxel_y, voxel_z)
        if voxel not in voxel_grid:
            voxel_grid[voxel] = [point]
        else:
            voxel_grid[voxel].append(point)
    voxel_centers = np.array(list(voxel_grid.keys())) * voxel_size + voxel_size / 2
    new_points = []
    new_colors = []
    for voxel, points in voxel_grid.items():
        new_points.append(np.mean(points, axis=0))
        new_colors.append(np.mean(colors[points], axis=0))
    return np.array(new_points), np.array(new_colors)



def filter_depth(scan_folder, out_folder, plyfilename):
    # # the pair file
    # pair_file = os.path.join(scan_folder, "pair.txt")
    # # for the final point cloud
    vertexs = []
    vertex_colors = []

    # pair_data = read_pair_file(pair_file)
    # nviews = len(pair_data)
    # TODO: hardcode size
    # used_mask = [np.zeros([296, 400], dtype=np.bool) for _ in range(nviews)]

    image_names = os.listdir(os.path.join(scan_folder, '3_views/images'))



    # for each reference view and the corresponding source views
    for ref_view in image_names:
        src_views = image_names.copy()
        src_views.remove(ref_view)
        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, f'cams/{ref_view[:-4]}_cam.txt'))
        # load the reference image
        ref_img = read_img(os.path.join(scan_folder, f'images/{ref_view}'))
        # load the estimated depth of the reference view
        # ref_depth_est = np.load(os.path.join(out_folder, f'depth_adjust_maps/depth_{ref_view[:-4]}.npy'))
        # ref_depth_est = np.load(os.path.join(out_folder, f'depth_adjust_maps_stereo/depth_{ref_view[:-4]}.npy'))
        
        ref_monodepth = read_pfm(os.path.join(out_folder, f'depth_maps/depth_{ref_view[:-4]}.pfm'))[0]
        depth_stereo_dir = os.path.join(scan_folder, '3_views')
        sparse_depth_map, sparse_points_map, intr, train_images_rt = read_stereo_sparse_points(depth_stereo_dir)

        sparse_depth_map_ref = sparse_depth_map[ref_view[:-4]]
        #close-form
        import cv2
        ref_monodepth = ref_monodepth.max() - ref_monodepth
        ref_monodepth = cv2.resize(ref_monodepth, (sparse_depth_map_ref.shape[1], sparse_depth_map_ref.shape[0]), interpolation=cv2.INTER_NEAREST)
        predict_depth_stereo, a, b = compute_scale_and_shift(ref_monodepth[sparse_depth_map_ref>0].reshape(-1, 1), sparse_depth_map_ref[sparse_depth_map_ref>0].reshape(-1, 1))
        ref_depth_est = a * ref_monodepth + b


        # # load the photometric mask of the reference view
        # confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        # photo_mask = confidence > 0.8
        photo_mask = np.ones_like(ref_depth_est, dtype=bool)

        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        # compute the geometric mask
        geo_mask_sum = 0
        for src_view in src_views:
            # camera parameters of the source view
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(scan_folder, f'cams/{src_view[:-4]}_cam.txt'))
            # the estimated depth of the source view
            
            # src_depth_est = np.load(os.path.join(out_folder, f'depth_adjust_maps_stereo/depth_{src_view[:-4]}.npy'))

            src_monodepth = read_pfm(os.path.join(out_folder, f'depth_maps/depth_{ref_view[:-4]}.pfm'))[0]
            sparse_depth_map_src = sparse_depth_map[src_view[:-4]]
            src_monodepth = src_monodepth.max() - src_monodepth
            src_monodepth = cv2.resize(src_monodepth, (sparse_depth_map_src.shape[1], sparse_depth_map_src.shape[0]), interpolation=cv2.INTER_NEAREST)
            predict_depth_stereo, a, b = compute_scale_and_shift(src_monodepth[sparse_depth_map_src>0].reshape(-1, 1), sparse_depth_map_src[sparse_depth_map_src>0].reshape(-1, 1))
            src_depth_est = a * src_monodepth + b



            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                      src_depth_est,
                                                                      src_intrinsics, src_extrinsics)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        # at least 3 source views matched
        geo_mask = geo_mask_sum >= 1
        final_mask = np.logical_and(photo_mask, geo_mask)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(scan_folder, ref_view,
                                                                                    photo_mask.mean(),
                                                                                    geo_mask.mean(), final_mask.mean()))

        if args.display:
            import cv2
            cv2.imshow('ref_img', ref_img[:, :, ::-1])
            cv2.imshow('ref_depth', ref_depth_est / 800)
            cv2.imshow('ref_depth * photo_mask', ref_depth_est * photo_mask.astype(np.float32) / 800)
            cv2.imshow('ref_depth * geo_mask', ref_depth_est * geo_mask.astype(np.float32) / 800)
            cv2.imshow('ref_depth * mask', ref_depth_est * final_mask.astype(np.float32) / 800)
            cv2.waitKey(0)

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        # valid_points = np.logical_and(final_mask, ~used_mask[ref_view])
        valid_points = final_mask
        print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        
        color = ref_img[valid_points]  # hardcoded for DTU dataset [1:-16:4, 1::4, :]
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

        # # 存储depth_est_averaged
        # depth_est_averaged[~valid_points] = 0
        # np.save(os.path.join(out_folder, f'depth_adjust_maps/depth_{ref_view[:-4]}_fusion.npy'), depth_est_averaged)
        # plt.imsave(os.path.join(out_folder, f'depth_adjust_maps/depth_{ref_view[:-4]}_fusion.jpg'), depth_est_averaged, cmap="jet")



        # # set used_mask[ref_view]
        # used_mask[ref_view][...] = True
        # for idx, src_view in enumerate(src_views):
        #     src_mask = np.logical_and(final_mask, all_srcview_geomask[idx])
        #     src_y = all_srcview_y[idx].astype(np.int)
        #     src_x = all_srcview_x[idx].astype(np.int)
        #     used_mask[src_view][src_y[src_mask], src_x[src_mask]] = True

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)

    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)




if __name__ == '__main__':
    # step1. save all the depth maps and the masks in outputs directory
    # save_depth()

    # with open(args.testlist) as f:
    #     scans = f.readlines()
    #     scans = [line.rstrip() for line in scans]

    # for scan in scans:
    #     scan_id = int(scan[4:])
    #     scan_folder = os.path.join(args.testpath, scan)
    #     out_folder = os.path.join(args.outdir, scan)
        # step2. filter saved depth maps with photometric confidence maps and geometric constraints
    scan_folder = '/home/dengyangyan/code/data/nerf_llff_data/flower'
    out_folder = '/home/dengyangyan/code/data/nerf_llff_data/flower'
    filter_depth(scan_folder, out_folder, os.path.join(args.outdir, 'mvsnet_close.ply'))

    # step3. 点云降采样
    import open3d as o3d
 
    # 读取文件
    pcd = o3d.io.read_point_cloud(os.path.join(args.outdir, 'mvsnet_close.ply'))  # path为文件路径
    pcd_new = o3d.geometry.PointCloud.uniform_down_sample(pcd, 1000)
    o3d.io.write_point_cloud(os.path.join(args.outdir, 'mvsnet_downsample.ply'), pcd_new)