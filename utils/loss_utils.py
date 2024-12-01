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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchmetrics.functional.regression import pearson_corrcoef
# from torch_scatter import scatter
import cv2
from torch import nn

def normalize_1(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=-1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=-1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))

def tqc_from_depth(depth_mono, intr, extrinsics, src_extrinsics):
    # colmap 的外参为c2w，需要转换为w2c!!!!
    # 计算tqc
    depth_mono_flatten = depth_mono.view(-1, 1)
    _, h, w = depth_mono.shape

    y, x = torch.meshgrid([torch.arange(0, h, dtype=torch.float32, device=intr.device),
                            torch.arange(0, w, dtype=torch.float32, device=intr.device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(h * w), x.view(h * w)
    uv = torch.stack((x, y, torch.ones_like(x))).T  # [3, H*W]


    # u = torch.meshgrid(torch.arange(w), torch.arange(h))[0].T.reshape(-1, 1).cuda(intr.device)
    # v = torch.meshgrid(torch.arange(w), torch.arange(h))[1].T.reshape(-1, 1).cuda(intr.device)
    # uv = torch.cat([u, v, torch.ones_like(u)], dim=1).float()

    # for debug
    Rot, Rot_src = extrinsics[0].T, src_extrinsics[0].T
    # Trans, Trans_src = Rot @ extrinsics[1].reshape(-1, 1), Rot_src @ src_extrinsics[1].reshape(-1, 1)
    # Rot, Rot_src = extrinsics[0], src_extrinsics[0]
    Trans, Trans_src = extrinsics[1].reshape(-1, 1), src_extrinsics[1].reshape(-1, 1)


    proj = (intr @ Rot_src) @ torch.linalg.inv(intr @ Rot)

    # 计算t
    t = (proj @ uv.T).T * depth_mono_flatten
    # 计算q
    q = (proj @ uv.T).T
    # 计算c
    c = (-intr @ Rot_src @ torch.linalg.inv(Rot) @ Trans + intr @ Trans_src).T
    c = c.repeat(h*w, 1)
    
    return t, q, c
def remap_with_pytorch(src, map1, map2, nview, mask=None):
    src = src.unsqueeze(0)# NCHW
    map1, map2 = map1.unsqueeze(0).unsqueeze(0), map2.unsqueeze(0).unsqueeze(0)
    # src = src.permute(0, 3, 1, 2)  

    H, W = src.shape[2:]
    nview, seg_cnt= map1.shape[:2]
    map1 = map1.view(nview*seg_cnt, H, W)
    map2 = map2.view(nview*seg_cnt, H, W)
    src = src.repeat(seg_cnt, 1, 1, 1)  # nview*seg, CHW


    map1_norm = (map1 / (W - 1)) * 2 - 1  # 归一化到[-1, 1]
    map2_norm = (map2 / (H - 1)) * 2 - 1  # 归一化到[-1, 1]
    grid = torch.stack([map1_norm, map2_norm], dim=-1)  # nview, seg, h, w, 2

    # grid中包含像素坐标的次数累加

    # grid_cnt = torch.zeros_like(grid)
    # pixels = torch.stack([map1, map2], dim=-1).squeeze()
    # pixels = pixels.int()
    # x' = grid_cnt[pixels]

    

    # 使用grid_sample进行重映射
    remapped_tensor = F.grid_sample(src, grid.float(), mode='bilinear', padding_mode='zeros')

    remapped_tensor = remapped_tensor.view(nview, 3, H, W)  # nview, seg, CHW


    return remapped_tensor

def seg_norm_mse_loss(input, target, seg, margin, return_mask=False):
    # seg = seg.unsqueeze(0)
    seg_idx = torch.unique(seg)
    loss = 0
    for i in seg_idx:
        input_patches = normalize_1(input[seg == i])
        target_patches = normalize_1(target[seg == i])
        
        loss += 1 - pearson_corrcoef(input_patches, -target_patches)
        # margin_l2_loss(input_patches, target_patches, margin, return_mask)
    return loss/len(seg_idx)

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_loss_mask(network_output, gt, mask=None):
    if mask is None:
        return l1_loss(network_output, gt)
    else:
        return torch.abs((network_output - gt) * mask).sum()/mask.sum()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, mask=None, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if mask is not None:
        img1 = img1 * mask + (1-mask)
        img2 = img2 * mask + (1-mask)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))

def normalize_seg(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=0, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=0, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))


def patchify(input, patch_size):
    patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0,2,1).view(-1, 1*patch_size*patch_size)
    return patches

def margin_l2_loss(network_output, gt, margin, return_mask=False):
    mask = (network_output - gt).abs() > margin
    if not return_mask:
        return ((network_output - gt)[mask] ** 2).mean()
    else:
        return ((network_output - gt)[mask] ** 2).mean(), mask
    
def patch_norm_mse_loss(input, target, patch_size, margin, return_mask=False):
    input_patches = normalize(patchify(input, patch_size))
    target_patches = normalize(patchify(target, patch_size))
    return margin_l2_loss(input_patches, target_patches, margin, return_mask)

def loss_depth_smoothness(depth, img):
    img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    weight_x = torch.exp(-torch.abs(img_grad_x).mean(1).unsqueeze(1))
    weight_y = torch.exp(-torch.abs(img_grad_y).mean(1).unsqueeze(1))

    loss = (((depth[:, :, :, :-1] - depth[:, :, :, 1:]).abs() * weight_x).sum() +
            ((depth[:, :, :-1, :] - depth[:, :, 1:, :]).abs() * weight_y).sum()) / \
           (weight_x.sum() + weight_y.sum())
    return loss

def patch_norm_mse_loss_global(input, target, patch_size, margin, return_mask=False):
    input_patches = normalize(patchify(input, patch_size), std = input.std().detach())
    target_patches = normalize(patchify(target, patch_size), std = target.std().detach())
    return margin_l2_loss(input_patches, target_patches, margin, return_mask)

def loss_photometric(image, gt_image, opt, valid=None):
    Ll1 =  l1_loss_mask(image, gt_image, mask=valid)
    loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image, mask=valid)))
    return loss

def _diffs(y):#y shape(bs, nfeat, vol_shape)
    ndims = y.ndimension() - 2
    df = [None] * ndims
    for i in range(ndims):
        d = i + 2#y shape(bs, c, d, h, w)
        # permute dimensions to put the ith dimension first
#            r = [d, *range(d), *range(d + 1, ndims + 2)]
        y = y.permute(d, *range(d), *range(d + 1, ndims + 2))
        
        dfi = y[1:, ...] - y[:-1, ...]
        
        # permute backlao
        # note: this might not be necessary for this loss specifically,
        # since the results are just summed over anyway.
#            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
        df[i] = dfi.permute(*range(1, d + 1), 0, *range(d + 1, ndims + 2))
        y = y.permute(*range(1, d + 1), 0, *range(d + 1, ndims + 2))
    
    return df

def penalty_loss(pred, penalty = 'l2'):
    # start_time = time.time()
    pred = pred.permute(2, 0, 1)
    pred = pred.unsqueeze(0)
    ndims = pred.ndimension() - 2
    df = 0
    diff_pred = _diffs(pred)

    for f in diff_pred:
        
        if penalty == 'l1':
            df += f.abs().mean() / ndims
        else:
            assert penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % penalty
            df += f.pow(2).mean() / ndims

    return df


def loss_feature_metric(language_feature, gt_language_feature, gt_unique_objects_all, seg_map, op, bg_mask=None):
    
    seg_map = seg_map.unsqueeze(0)

    gt_language_feature = gt_language_feature.permute(1, 2, 0)
    language_feature = language_feature.permute(1, 2, 0)

    # 计算language_feature和language_feature_map 的余弦相似度
    language_feature_new = language_feature.view(-1, 3)
    z_pred = F.cosine_similarity(language_feature_new.unsqueeze(1), gt_unique_objects_all.unsqueeze(0), dim=2)
    p_k = torch.exp(z_pred) / torch.exp(z_pred).sum(dim=1).unsqueeze(1)
    

    # 计算gt_language_feature和language_feature_map的余弦相似度
    gt_language_feature_new = gt_language_feature.view(-1, 3)
    z_gt = F.cosine_similarity(gt_language_feature_new.unsqueeze(1), gt_unique_objects_all.unsqueeze(0), dim=2)
    

    seg_map_1d = seg_map.view(-1)
    one_hot_seg = torch.zeros_like(z_gt)
    one_hot_seg.scatter_(1, seg_map_1d.unsqueeze(1), 1)

    epsilon = 1e-3

    q_k = (1-epsilon) * one_hot_seg + epsilon / gt_unique_objects_all.shape[0]

    # print(q_k)

    # 计算交叉熵损失
    if bg_mask is not None:
        loss_feature_known_ce = op.known_fce * (-torch.sum(q_k[~bg_mask.view(-1)] * torch.log(p_k[~bg_mask.view(-1)] + 1e-8), dim=1).mean())
    else:
        loss_feature_known_ce = op.known_fce * (-torch.sum(q_k * torch.log(p_k + 1e-8), dim=1).mean())
    # print(loss_feature_known_ce)
    
    # p_k_gt = torch.exp(z_gt) / torch.exp(z_gt).sum(dim=1).unsqueeze(1) # 有问题



    
    # 计算交叉熵损失
    # loss_feature_known_ce = op.known_fce * (-torch.sum(p_k_gt * torch.log(p_k + 1e-8), dim=1).mean())          

    loss_feature_l1 = op.known_fl1 * l1_loss(language_feature_new, gt_language_feature_new) # 1

    loss_feature = loss_feature_known_ce  + loss_feature_l1

    # loss_feature = loss_feature_l1

    """
    平滑损失函数
    """
    # smoothness loss
    smooth_loss = penalty_loss(language_feature, penalty='l2')
    loss_feature_smooth_known =  op.known_fsm * smooth_loss# 0.000001

    return loss_feature, loss_feature_smooth_known

def loss_depth_metric(depth, depth_mono):
    depth_mono = depth_mono.reshape(-1, 1)
    depth = depth.reshape(-1, 1)[depth_mono>0]
    depth_mono = depth_mono[depth_mono>0]

    depth_loss = min(
                    (1 - pearson_corrcoef(depth_mono, depth)),
                    (1 - pearson_corrcoef(1 / (-depth_mono + 100), depth))
    ) 

    return depth_loss

def loss_reproject_depth(rendered_depth_pseudo, scene, pseudo_cam, consistency_view_thresh=2, nviews=3):
    h, w = rendered_depth_pseudo.shape[-2:]

    depth_hecheng = torch.zeros((nviews, h, w), device=rendered_depth_pseudo.device)
    # cv2.imwrite("pseudo_image.jpg", pseudo_image.permute(1, 2, 0).detach().cpu().numpy()*255)
    with torch.no_grad():
        # 对depth_hecheng进行处理，取通道差异最小两值的均值
        for idx, ref_view in enumerate(scene.getTrainCameras().copy()):

            intr = ref_view.intrinsics[:3, :3]
            ref_extrinsics = [torch.from_numpy(ref_view.R).cuda(intr.device).float(), torch.from_numpy(ref_view.T).cuda(intr.device).float()]
            src_extrinsics = [torch.from_numpy(pseudo_cam.R).cuda(intr.device).float(), torch.from_numpy(pseudo_cam.T).cuda(intr.device).float()]
            t, q, c = tqc_from_depth(ref_view.depth_mono, intr, ref_extrinsics, src_extrinsics)
            x_new = t + c
            u_new = (x_new[:, 0]/(x_new[:, 2])).round().to(torch.int32)
            v_new = (x_new[:, 1]/(x_new[:, 2])).round().to(torch.int32)
            depth_new = x_new[:, 2]
            depth_adjust = torch.zeros_like(depth_new).view(-1, 1)
            valid_mask = (u_new >= 0) & (u_new < w) & (v_new >= 0) & (v_new < h)
            depth_new[~valid_mask] = 0
            u_new[u_new < 0] = 0
            v_new[v_new < 0] = 0
            u_new[u_new >= w] = 0
            v_new[v_new >= h] = 0

            index = (u_new + v_new*(w)).to(torch.int64)

            depth_adjust = scatter(depth_new, index, dim=0, reduce="min")
            depth_adjust = torch.cat([depth_adjust, torch.zeros((h*w - depth_adjust.shape[0]), device=depth_adjust.device)], dim=0)

            depth_adjust = depth_adjust.view(h, w)

            depth_hecheng[idx, :, :] = depth_adjust
            # # 可视化展示depth_new
            # depth_new_show = depth_adjust.detach().cpu().numpy()
            # cv2.imwrite("depth_new.jpg", depth_new_show)

        nviews = depth_hecheng.shape[0]
        depth_zeros_cnt = (depth_hecheng == 0).sum(dim=0)
        depth_pesudo = (depth_hecheng.sum(dim=0) / (nviews - depth_zeros_cnt + 1e-6)).detach()

        error_map = (depth_hecheng - depth_pesudo.unsqueeze(0)).abs()
        # depth_pesudo_range = ((depth_pesudo[(depth_pesudo<1000)&(depth_pesudo>0)].max() - depth_pesudo[(depth_pesudo<1000)&(depth_pesudo>0)].min())*0.005)#0.01   ("trex" "flower" "fortress") 0.005
        depth_pesudo_range = 0.05#0.01   dtu 

        # print(depth_pesudo[(depth_pesudo<10000)&(depth_pesudo>0)].max(), depth_pesudo[(depth_pesudo<10000)&(depth_pesudo>0)].min())
        # print(depth_pesudo_range)
        error_map_cnt = (error_map < depth_pesudo_range).sum(dim=0)
        valid_mask = (error_map_cnt >= consistency_view_thresh)
        depth_pesudo[~valid_mask] = 0
        depth_pesudo = depth_pesudo.detach()

        depth_new_show = depth_pesudo.cpu().numpy()
        cv2.imwrite("depth_new.jpg", depth_new_show)

    # L2损失
    # depth_loss_pseudo_v2 = 1 * l2_loss(rendered_depth_pseudo[depth_pesudo>0], depth_pesudo[depth_pesudo>0]) #fortress
    depth_loss_pseudo_v2 = 0.5 * min(
                    (1 - pearson_corrcoef(depth_pesudo[depth_pesudo>0], rendered_depth_pseudo[depth_pesudo>0])),
                    (1 - pearson_corrcoef(1 / (-depth_pesudo[depth_pesudo>0] + 200), rendered_depth_pseudo[depth_pesudo>0]))
                    
    )

    return depth_loss_pseudo_v2
