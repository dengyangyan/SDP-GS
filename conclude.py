import os
import argparse
from compare_llff import read_pfm, read_stereo_sparse_points, compute_scale_and_shift, create_grid_seg
import numpy as np
import cv2
from sklearn.linear_model import (LinearRegression, HuberRegressor, RANSACRegressor, TheilSenRegressor)
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from torchmetrics.functional.regression import pearson_corrcoef
import torch
import shutil
import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root_path', type=str, default="/home/dengyangyan/code/experiment/FSGS-main_total/semantic-aware-GS/nerf_llff_data")
args = parser.parse_args()
def get_boundary_pixels(matrix, region, neighbors_segidxes):
    """
    获取图像中指定区域的边缘点

    参数:
    matrix (np.ndarray): 输入图像矩阵
    region (np.ndarray): 布尔矩阵，指定区域为True，其余为False

    返回:
    List[Tuple[int, int]]: 边缘点的坐标列表
    """
    rows, cols = matrix.shape[:2]
    # 将布尔矩阵转换为uint8类型
    region_uint8 = region.astype(np.uint8) * 255

    # 使用Sobel算子进行边缘检测
    sobelx = cv2.Sobel(region_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(region_uint8, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.hypot(sobelx, sobely)

    # 找到边缘点的坐标
    boundary_pixels = np.argwhere(sobel > 0)

    for pixel in boundary_pixels:
        x, y = pixel
        if region[x, y]:
            # 检查四个邻居是否在区域外
            if x > 0 and not region[x-1, y] and matrix[x-1, y] not in neighbors_segidxes:
                neighbors_segidxes.append(matrix[x-1, y])
            if x < rows-1 and not region[x+1, y] and matrix[x+1, y] not in neighbors_segidxes:
                neighbors_segidxes.append(matrix[x+1, y])
            if y > 0 and not region[x, y-1] and matrix[x, y-1] not in neighbors_segidxes:
                neighbors_segidxes.append(matrix[x, y-1])
            if y < cols-1 and not region[x, y+1] and matrix[x, y+1] not in neighbors_segidxes:
                neighbors_segidxes.append(matrix[x, y+1])

    return neighbors_segidxes

# 计算拟合不同segment的depth_mono与depth_stereo的相关性计算结果
def conclude_depth(depth_mono_ori, depth_stereo, seg_ori, image, colors=None, scenes=None, image_name=None, depth_adjust_file=None, error_map_file=None):
    # 计算depth_mono与depth_stereo的相关性
    seg_idx_unique = [int(seg_idx) for seg_idx in np.unique(seg_ori)]
    valid_mask = depth_stereo > 0
    robust_num = 20 # 40
    image = cv2.resize(image, (depth_stereo.shape[1], depth_stereo.shape[0]))# 缩放图像与depth_stereo一样大
    seg = cv2.resize(seg_ori, (depth_stereo.shape[1], depth_stereo.shape[0]), interpolation=cv2.INTER_NEAREST)

    depth_mono = cv2.resize(depth_mono_ori, (depth_stereo.shape[1], depth_stereo.shape[0]), interpolation=cv2.INTER_NEAREST)

    if valid_mask.sum() == 0:
        plt.imsave(depth_adjust_file.replace('.npy', '_stereo.jpg'), depth_stereo, cmap="gray")
        np.save(depth_adjust_file, depth_mono.max()-depth_mono)
        plt.imsave(depth_adjust_file.replace('.npy', '_adjust.jpg'), depth_mono.max()-depth_mono, cmap="gray")
        plt.imsave(depth_adjust_file.replace('.npy', '_mono.jpg'), depth_mono, cmap="gray")

    else:

        ymin, ymax = 0, depth_stereo[valid_mask].max()
        plotline_X = np.arange(depth_mono[valid_mask].min(), depth_mono[valid_mask].max(), 0.1).reshape(-1, 1)
            
        linear_Ab = {}

        for seg_idx in seg_idx_unique:
            seg_mask = seg == seg_idx
            depth_mono_seg = depth_mono[valid_mask&seg_mask].reshape(-1, 1)
            depth_stereo_seg = depth_stereo[valid_mask&seg_mask].reshape(-1, 1)

            
            if depth_stereo_seg.shape[0] >= robust_num:
                # 计算相关性
                """
                ransac 方法
                """
                ransac = RANSACRegressor(min_samples=0.7, max_trials=500, random_state=10).fit(depth_mono_seg, depth_stereo_seg)
                
                predict_y = ransac.predict(plotline_X)
                predict_mono = ransac.predict(depth_mono_seg)
                ransac_coef = ransac.estimator_.coef_

                # print(f"{seg_idx} ransac_regression : ", ransac.estimator_.coef_[0])

                a = (predict_y[-1] - predict_y[0]) / (plotline_X[-1] - plotline_X[0])
                b = predict_y[0] - a * plotline_X[0]
                linear_Ab[seg_idx] = [a, b]

            

        predict_depth_stereo, total_a, total_b = compute_scale_and_shift(depth_mono[valid_mask].reshape(-1, 1), depth_stereo[valid_mask].reshape(-1, 1))


        if linear_Ab == {}:
            linear_Ab[0] = np.array([1, 0]).reshape(2, 1)
        
        for seg_idx in seg_idx_unique:
            seg_mask = seg == seg_idx
            depth_mono_seg = depth_mono[valid_mask&seg_mask].reshape(-1, 1)
            depth_stereo_seg = depth_stereo[valid_mask&seg_mask].reshape(-1, 1)
            
            if depth_stereo_seg.shape[0] < robust_num:
                #连通域标记
                seg_mask_uint8 = seg_mask.astype(np.uint8)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(seg_mask_uint8, connectivity=8)
                # 在每一个连通域中找到邻接的连通域
                neighbors_segidxes= []
                for label in range(1, num_labels):
                    region = labels == label
                    if region.sum() < 1000:
                        continue
                    neighbors_segidxes = get_boundary_pixels(seg, region, neighbors_segidxes)

                    # neighbors_segidxes.extend(neighbors_segidx)
            
                # cv2.imwrite("seg.jpg", (seg==seg_idx) * 255)
                # seg_neighbor = np.zeros_like(seg, dtype=bool)
                # for seg_idx_neigh in neighbors_segidxes:
                #     seg_neighbor = seg_neighbor | (seg == seg_idx_neigh)
                # cv2.imwrite("seg_neighbor.jpg", seg_neighbor*255)

                neighbors_segidxes = sorted([seg_idx_neigh for seg_idx_neigh in neighbors_segidxes], key=lambda x: (seg==x).sum(), reverse=True)

                for seg_idx_neigh in neighbors_segidxes:
                    if linear_Ab.get(seg_idx_neigh) is not None:
                        linear_Ab[seg_idx] = linear_Ab[seg_idx_neigh]
                        break
                
                if seg_idx not in linear_Ab.keys():
                    linear_Ab[seg_idx] = [np.array([total_a]), np.array([total_b])]

                
                if depth_stereo_seg.shape[0] ==0 :
                    continue
                else:
                    r_2_min = 1000
                    for key in linear_Ab.keys():
                        # 直线为y=ax+b，斜率为a，截距为b
                        a, b = linear_Ab[key]
                        # 计算当前每个点相对于拟合直线的距离
                        r_2 = abs(depth_stereo_seg - a*depth_mono_seg - b)/np.sqrt(a**2 + 1)

                        r_2_mean = r_2.mean()
                        
                        if r_2_mean < r_2_min:
                            r_2_min = r_2_mean
                            linear_Ab[seg_idx] = [a, b]
                        
        if not os.path.exists(error_map_file):
            os.makedirs(error_map_file)


        # 热力图可视化
        """
        colse form 方法
        """
        predict_depth_stereo, a, b = compute_scale_and_shift(depth_mono[valid_mask].reshape(-1, 1), depth_stereo[valid_mask].reshape(-1, 1))
        predict_y = a * plotline_X + b
        predict_depth_stereo = a * depth_mono + b
        error_map_all = np.zeros_like(depth_mono)
        
        error_map_all[:, :] = abs(depth_stereo[valid_mask] - predict_depth_stereo[valid_mask]).mean()
        error_max = error_map_all.max()

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # plt.figure(figsize=(16, 9))
        # plt.rc('font',family='Times New Roman', size=20)
        # norm = matplotlib.colors.Normalize(vmin=0, vmax=error_max)    
        # plt.subplot(1, 2, 1)
        # plt.imshow(image)
        # plt.imshow(error_map_all, cmap="jet", norm=norm, alpha=0.7)
        # plt.colorbar(shrink=0.5)
        # plt.axis('off')
        # plt.title("error map whole")
           
        # depth_adjust = np.zeros_like(depth_mono)
        # error_map = np.zeros_like(depth_mono)
        # for seg_idx in seg_idx_unique:
        #     seg_mask = seg == seg_idx
        #     a = linear_Ab[seg_idx][0][0]
        #     b = linear_Ab[seg_idx][1][0]
        #     if (seg_mask&valid_mask).sum() == 0:
        #         continue
        #     error_map[seg_mask] = abs(depth_stereo[seg_mask&valid_mask] - a*depth_mono[seg_mask&valid_mask] - b).mean()

        # plt.subplot(1, 2, 2)
        # plt.imshow(image)
        # plt.imshow(error_map, cmap="jet", norm=norm, alpha=0.7)
        # plt.colorbar(shrink=0.5)
        # plt.axis('off')
        # plt.title("error map seg")
        # plt.savefig(f"{error_map_file}/{image_name}.jpg")   


        


        # 子图逐点可视化
        unique_abs = np.unique(np.array(list(linear_Ab.values()), dtype=np.float32), axis=0)
        colors = {tuple(unique_abs[i, :, 0]): np.random.rand(3) for i in range(unique_abs.shape[0])}  # ransac
        # colors = {tuple(unique_abs[i, :]): np.random.rand(3) for i in range(unique_abs.shape[0])}  # close form
        # colors转变为float32
        colors = {tuple(np.array(key, dtype=np.float32)): value for key, value in colors.items()}


        
        # 可视化拟合直线与散点图
        n_cols = 4
        n_rows = len(unique_abs) // n_cols + 1
        fig, ax = plt.subplots(n_rows, 2*n_cols, figsize=(36, 16))
        plt.rc('font',family='Times New Roman', size=12)
        # 设置子图间距
        plt.subplots_adjust(left=0, right=0.98, top=0.95, bottom=0.05, wspace=0.25, hspace=0.5)
        
        # 确保ax总是二维的
        if n_rows == 1 or 2*n_cols == 1:
            ax = np.array([ax])

        for cnts, ab in enumerate(unique_abs):
            SEGS_name = []
            a, b = ab
            image_SEG = image.copy()
            alpha = 0.8
            # color_seg = colors[tuple(ab[:2])]# close form
            color_ = colors[tuple(ab[:2,0])]# ransac
            
            color_seg = np.ones(3, dtype=np.uint8) * 255 # if color_seg is None else color_seg
            valid_mask_seg = np.zeros_like(valid_mask, dtype=bool)
            mask_image = np.zeros_like(seg, dtype=bool)
            
            for seg_idx in seg_idx_unique:
                seg_a, seg_b = linear_Ab[seg_idx]
                if abs(a - seg_a)<1e-4 and abs(b-seg_b)<1e-4:
                    seg_mask = seg == seg_idx
                    valid_mask_seg = valid_mask_seg | (valid_mask & seg_mask)                
                    mask_image = mask_image | seg_mask
                    SEGS_name.append(seg_idx)

            mask_image_seg = np.stack([mask_image]*3, axis=-1)
            image_SEG = image_SEG[:, :, ::-1 ]
            image_SEG = np.where(mask_image_seg, image_SEG * (1 - alpha) + color_seg * alpha, image_SEG)
            image_SEG = image_SEG.astype(np.uint8)  # 保证图像数据类型为uint8

            depth_mono_seg = depth_mono[valid_mask_seg].reshape(-1, 1)
            depth_stereo_seg = depth_stereo[valid_mask_seg].reshape(-1, 1)
            predict_y = a * plotline_X + b

            r_2 = (abs(depth_stereo_seg - a*depth_mono_seg - b)/np.sqrt(a**2 + 1))
            r_2_mean = r_2.mean()

            n_row = cnts // n_cols
            n_col = cnts % n_cols
            ax[n_row, 2*n_col].imshow(image_SEG)
            ax[n_row, 2*n_col].axis('off')
            ax[n_row, 2*n_col+1].scatter(depth_mono_seg, depth_stereo_seg, color=color_, label='inlier', s=0.2)
            ax[n_row, 2*n_col+1].plot(plotline_X, predict_y, color=color_, label='ransac')
            ax[n_row, 2*n_col+1].set_ylim(ymin, ymax)
            # ax[n_row, 2*n_col+1].set_ylim(ymin, 60)
            ax[n_row, 2*n_col+1].set_xlabel('mono depth')
            ax[n_row, 2*n_col+1].set_ylabel('nearly GT depth')

            ax[n_row, 2*n_col+1].spines['right'].set_visible(False)
            ax[n_row, 2*n_col+1].spines['top'].set_visible(False)
        
        for cnts in range(len(unique_abs), n_cols*n_rows):
            n_row = cnts // n_cols
            n_col = cnts % n_cols
            ax[n_row, 2*n_col].axis('off')
            ax[n_row, 2*n_col+1].axis('off')

        savename = depth_adjust_file.replace('.npy', '_ransac.jpg')
        plt.savefig(savename)
        plt.close()



        # 存储单目深度图的scale调整结果
        depth_adjust = np.zeros_like(depth_mono)
        if depth_adjust_file is not None:
            for seg_idx in seg_idx_unique:
                seg_mask = seg == seg_idx
                a = linear_Ab[seg_idx][0][0]
                b = linear_Ab[seg_idx][1][0]
                depth_adjust[seg_mask] = depth_mono[seg_mask] * a + b
            depth_adjust = cv2.resize(depth_adjust, (depth_stereo.shape[1], depth_stereo.shape[0]), interpolation=cv2.INTER_NEAREST)
            # plt.subplot(1, 2, 1)
            # plt.scatter(depth_mono[valid_mask], depth_stereo[valid_mask], color='b', label='inlier')

            # plt.subplot(1, 2, 2)
            # plt.scatter(depth_adjust[valid_mask], depth_stereo[valid_mask], color='r', label='predict')
            # plt.savefig(f"conclude_grid/{scenes}/{image_name}/depth_adjust.jpg")

            # # 计算皮尔逊相关系数
            # pearson_before = stats.pearsonr(depth_mono[valid_mask].reshape(-1), depth_stereo[valid_mask].reshape(-1))
            # pearson = stats.pearsonr(depth_adjust[valid_mask].reshape(-1), depth_stereo[valid_mask].reshape(-1))
            # print(f"scene {scenes} image {image_name} pearson before: {pearson_before} after: {pearson}")
            # with open(f"conclude_grid/{scenes}/pearson.txt", "a+") as f:
            #     f.write(f"scene {scenes} image {image_name} pearson before: {pearson_before} after: {pearson} \n")

            np.save(depth_adjust_file, depth_adjust)
            # # 存储可视化展示

            plt.imsave(depth_adjust_file.replace('.npy', '_stereo.jpg'), depth_stereo, cmap="gray")
            plt.imsave(depth_adjust_file.replace('.npy', '_adjust.jpg'), depth_adjust, cmap="gray")
            plt.imsave(depth_adjust_file.replace('.npy', '_mono.jpg'), depth_mono, cmap="gray")

            
                
                    
        
                    




if __name__ == "__main__":
    scenes = os.listdir(args.root_path)

    for scene_id in scenes:
        scene_dir = os.path.join(args.root_path, scene_id)

        views = ['3_views']

        # stereo path
        depth_stereo_dir = os.path.join(scene_dir, views[0])
        sparse_depth_map = {}
        if not os.path.exists(os.path.join(depth_stereo_dir, 'dense/fused.ply')):
            depth_adjust_dir = os.path.join(scene_dir, "depth_adjust_maps_stereo_anything")
            if not os.path.exists(depth_adjust_dir):
                os.makedirs(depth_adjust_dir)
            for image_name in os.listdir(os.path.join(depth_stereo_dir, "images")):
                
                file_old = os.path.join(scene_dir, "depth_maps_anything", 'depth_' + image_name[:-4]+".pfm")

                depth_mono = read_pfm(file_old)[0]
                depth_mono = depth_mono.max() - depth_mono
                file_new = os.path.join(depth_adjust_dir, 'depth_' +  image_name[:-4]+".npy")
                np.save(file_new, depth_mono)
        else:
            sparse_depth_map, sparse_points_map, intr, train_images_rt = read_stereo_sparse_points(depth_stereo_dir)
            
            depth_mono_dir = os.path.join(scene_dir,"depth_maps_anything")
            segment_dir = os.path.join(scene_dir, "language_features_GGrouping_dim3")
            image_dir = os.path.join(scene_dir,"images")

            depth_adjust_dir = os.path.join(scene_dir,"depth_adjust_maps_stereo_anything")
            if not os.path.exists(depth_adjust_dir):
                os.makedirs(depth_adjust_dir)

            
            segment_files = [i for i in os.listdir(segment_dir) if i.endswith("_s.npy")]
            
            # 颜色分配
            seg_conclude = []
            for segment_file in segment_files:
                seg = np.load(os.path.join(segment_dir, segment_file))
                seg_idx_unique = np.unique(seg)
                seg_conclude.append(seg_idx_unique)
            seg_conclude = np.unique(np.concatenate(seg_conclude))
            colors = {seg_conclude[i]: (np.random.rand(3)*255).astype(np.int8) for i in range(len(seg_conclude))}


            
            for idx, seg_file in enumerate(segment_files):

                # depth_gt_file = os.path.join(depth_stereo_dir,seg_file.replace("_s.npy", ".npy"))
                depth_mono_file = os.path.join(depth_mono_dir, "depth_" + seg_file.replace("_s.npy", ".pfm")) # 单目深度先验
                depth_adjust_file = os.path.join(depth_adjust_dir, "depth_" + seg_file.replace("_s.npy", ".npy")) # 存储单目深度图的scale调整结果

                

                # output_show_file = os.path.join(output_show_path, seg_file.replace("_s.npy", ".jpg"))

                seg_file_path = os.path.join(segment_dir, seg_file)
                image_file = os.path.join(image_dir, seg_file.replace("_s.npy", ".JPG"))
                if not os.path.exists(image_file):
                    image_file = os.path.join(image_dir, seg_file.replace("_s.npy", ".png"))
                if not os.path.exists(image_file):
                    image_file = os.path.join(image_dir, seg_file.replace("_s.npy", ".jpg"))

                image = cv2.imread(image_file)
                h, w = image.shape[:2]

                depth_mono = read_pfm(depth_mono_file)[0]
                depth_mono = depth_mono.max() - depth_mono


                seg = np.load(seg_file_path)[0]

                depth_stereo = sparse_depth_map[seg_file.replace("_s.npy", "")]
                conclude_depth(depth_mono, depth_stereo, seg, image, colors=colors, scenes=scene_id,
                                image_name=seg_file.replace("_s.npy", ""),
                                depth_adjust_file=depth_adjust_file,
                                error_map_file=os.path.join(scene_dir, "error_map_stereo"))

                print(f"scene {scene_id} image {seg_file} conclude done")
