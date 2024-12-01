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
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args,OptimizationParams
from gaussian_renderer import GaussianModel
import cv2
import numpy as np
from utils.general_utils import vis_depth

def render_set(model_path, name, iteration, views, gaussians, pipeline, opt, background, args):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    render_feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(render_feature_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, opt)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        if args.render_depth:
            # depth_map = vis_depth(rendering['depth'][0].detach().cpu().numpy())
            depth_map = rendering['depth'][0].detach().cpu().numpy()
            np.save(os.path.join(render_depth_path, view.image_name + '_depth.npy'), rendering['depth'][0].detach().cpu().numpy())
            depth_map = (depth_map / depth_map.max() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(render_depth_path, view.image_name + '_depth.png'), depth_map)
        if args.render_alpha:
            alpha_map = (rendering['alpha'][0].detach().cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(render_path, view.image_name + '_alpha.png'), alpha_map)
        if opt.include_feature:
            print(rendering['feature'].shape)
            feature_map = rendering['feature']
            # h, w = feature_map.shape[1], feature_map.shape[2]

            # language_feature_new = feature_map.view(3, h*w).permute(1, 0)
            # z_pred = F.cosine_similarity(language_feature_new.unsqueeze(1), gt_unique_objects_all.unsqueeze(0), dim=2)
            # # p_k = torch.exp(z_pred) / torch.exp(z_pred).sum(dim=1).unsqueeze(1)
            # # 得到当前图像的相似度矩阵，获取最大相似度的物体
            # z_pred_max = torch.max(z_pred, dim=1)[0]
            # z_pred_mask = z_pred_max > 0.95
            # z_pred_id = -torch.ones(h*w, dtype=torch.int64).cuda()
            # z_pred_id[z_pred_mask] = torch.argmax(z_pred[z_pred_mask], dim=1)
            # z_pred_id = z_pred_id.detach().cpu().numpy()
            # # mask_id = -np.ones((h, w), dtype=np.int64)
            # mask_id = z_pred_id.reshape(h, w)
            # # 显示当前的mask id图像
            # mask_id_image = mask_id/gt_unique_objects_all.shape[0]*255
            # cv2.imwrite(os.path.join(render_path, view.image_name + '_mask_id.png'), mask_id_image)

            feature_map = (feature_map+1)/2
            feature_map = feature_map.permute(1, 2, 0).detach().cpu().numpy() * 255
            # feature_map = cv2.cvtColor(feature_map, cv2.COLOR_RGB2BGR)
            
            # torchvision.utils.save_image(rendering['language_feature_image'], view.image_name + '_feature.png')
            cv2.imwrite(os.path.join(render_feature_path, view.image_name + '_feature.png'), feature_map)
            # gt_language_feature_ori, language_feature_mask = view.get_language_feature_singlemasks(language_feature_dir=args.lf_path, feature_level=1)
            # gt_language_feature = (gt_language_feature_ori/gt_language_feature_ori.max()).detach().cpu().numpy() * 255
            # cv2.imwrite(os.path.join(gts_path, view.image_name + '_feature.png'), gt_language_feature[0])


def render_sets(dataset : ModelParams, opt: OptimizationParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if opt.include_feature:
            checkpoint = os.path.join(args.model_path, 'chkpnt'+str(args.iteration)+'.pth')
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, args, mode='test')

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, opt, background,args)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, opt, background,args)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=10000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_depth", action="store_true")
    parser.add_argument("--render_alpha", action="store_true")
    args = get_combined_args(parser)
    # print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), op.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)