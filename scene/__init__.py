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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, renderCameraList_from_camInfos
from utils.pose_utils import generate_random_poses_llff, generate_random_poses_360, generate_random_poses_llff_ours, generate_random_poses_blender
from scene.cameras import PseudoCamera
import torch, cv2
import numpy as np

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.pseudo_cameras = {}
        # if os.path.exists(os.path.join(args.source_path, "transforms.json")):
        #     print("Found transforms_train.json file, assuming Blender data set!")
        #     scene_info = sceneLoadTypeCallbacks["mipnerf360"](args.source_path, args.white_background, args.eval, n_views=args.nviews, features_name=args.language_features_name)

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.language_features_name, args.eval, n_views=args.nviews)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, n_views=args.nviews, features_name=args.language_features_name)

        else:
            print(args.source_path)
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(self.cameras_extent, 'cameras_extent')

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            """
            构建当前场景的feature_dict_gt
            """
            # 调整seg_id，获取不同视角segment id 对应的features
            feature_dict_gt = []
            seg_id_gt = []
            num_segments = 0
            for idx_view, cam in enumerate(self.train_cameras[resolution_scale]):
                language_feature, feature_dict, seg_map_ori = cam.point_feature, cam.feature_dict, cam.seg_map
                current_seg_id = seg_map_ori.unique()

                feature_dict_gt.append(feature_dict.tolist())
                seg_id_gt.append(current_seg_id.tolist())
                num_segments = max(num_segments, max(current_seg_id).item()+1)

            
            # 把seg_id_gt里面的id映射到0~num_segments            
            counts_seg_id = torch.zeros(int(num_segments)).to(language_feature.device)
            for seg_i in seg_id_gt:
                counts_seg_id[seg_i] += 1
            
            # 查找counts中不为0的索引
            seg_id_gts = torch.nonzero(counts_seg_id).cpu().numpy()
            counts_seg_id_new = counts_seg_id[seg_id_gts[:,0]]# 每一个segment的连续视角数目
            # mapping seg id
            seg_id_gts_mapping = {str(seg_id_gts[i,0]): i for i in range(seg_id_gts.shape[0])}
            num_segments = seg_id_gts.shape[0]
            
            feature_dict_gt_tensor = [None for i in range(args.nviews)]

            for idx, seg_id in enumerate(seg_id_gt):
                feature_dict_idx = np.zeros((num_segments, 3))
                for idx_seg, seg_id_per in enumerate(seg_id):
                    old_seg_id = int(seg_id_per)
                    new_seg_id = seg_id_gts_mapping[str(old_seg_id)]
                    feature_dict_idx[new_seg_id] = feature_dict_gt[idx][idx_seg]
                feature_dict_gt_tensor[idx] = feature_dict_idx

            feature_dict_gt_tensor = torch.tensor(feature_dict_gt_tensor, dtype=torch.float32, device=args.data_device)
            # 对于每一个segment，取对应的feature的均值
           
            feature_dict_gt_tensor = feature_dict_gt_tensor.sum(dim=0)/counts_seg_id_new.unsqueeze(1)

            for idx_view, cam in enumerate(self.train_cameras[resolution_scale]):            

                # 替换对应的feature
                language_feature, feature_dict, seg_map_ori = cam.point_feature, cam.feature_dict, cam.seg_map
                cv2.imwrite(f"{args.source_path}/language_features_GGrouping_dim3/fdim3_{cam.image_name}.jpg",  ((language_feature.permute(1, 2, 0)+1)/2*255).cpu().numpy())

                # 映射seg_id
                new_seg_map = torch.zeros_like(seg_map_ori).to(args.data_device)
                
                # 替换对应的feature
                new_language_feature = torch.zeros_like(language_feature).to(args.data_device)

                for i in np.unique(seg_map_ori.cpu().numpy()):
                    
                    new_seg_id = seg_id_gts_mapping[str(int(i))]
                    new_seg_map[seg_map_ori==i] = new_seg_id
                    
                    new_language_feature[:, seg_map_ori==i] = feature_dict_gt_tensor[new_seg_id].unsqueeze(1)

                self.train_cameras[resolution_scale][idx_view].seg_map = new_seg_map
                self.train_cameras[resolution_scale][idx_view].point_feature = new_language_feature
                self.train_cameras[resolution_scale][idx_view].feature_dict = feature_dict_gt_tensor
                # print("feature_dict_gt_tensor", feature_dict_gt_tensor)
            

                language_feature = cam.point_feature.permute(1, 2, 0)
                cv2.imwrite(f"{args.source_path}/language_features_GGrouping_dim3/language_adjust_{cam.image_name}.jpg",  ((language_feature+1)/2*255).cpu().numpy())

            
            
            # print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)


            pseudo_cams = []
            if args.source_path.find('llff')>=0:
                pseudo_poses = generate_random_poses_llff(self.train_cameras[resolution_scale])
            elif args.source_path.find('dtu')>=0:
                pseudo_poses = generate_random_poses_llff_ours(self.train_cameras[resolution_scale])
            elif args.source_path.find('360')>=0:
                pseudo_poses = generate_random_poses_360(self.train_cameras[resolution_scale])
            elif args.source_path.find('nerf_synthetic')>=0:
                pseudo_poses = generate_random_poses_360(self.train_cameras[resolution_scale])

            view = self.train_cameras[resolution_scale][0]
            for pose in pseudo_poses:
                pseudo_cams.append(PseudoCamera(
                    R=pose[:3, :3].T, T=pose[:3, 3], FoVx=view.FoVx, FoVy=view.FoVy,
                    width=view.image_width, height=view.image_height, intr=view.intrinsics
                ))
            self.pseudo_cameras[resolution_scale] = pseudo_cams





        self.init_point_cloud = scene_info.point_cloud
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getPseudoCameras(self, scale=1.0):
        if len(self.pseudo_cameras) == 0:
            return [None]
        else:
            return self.pseudo_cameras[scale]
        


class RenderScene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, spiral=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.test_cameras = {}

        if 'scan' in args.source_path:
            # scene_info = sceneLoadTypeCallbacks["SpiralDTU"](args.source_path)
            scene_info = sceneLoadTypeCallbacks["Spiral"](args.source_path)
        else:
            scene_info = sceneLoadTypeCallbacks["Spiral"](args.source_path)
        
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Render Cameras", resolution_scales)
            self.test_cameras[resolution_scale] = renderCameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            pass


    def getRenderCameras(self, scale=1.0):
        return self.test_cameras[scale]
