import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import argparse
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser()
# parser.add_argument('-b', '--benchmark', type=str) 
# parser.add_argument('-d', '--dataset_id', type=str)
parser.add_argument('-r', '--root_path', type=str)
args = parser.parse_args()
def compute_scale_and_shift(prediction, target):
    # print(prediction.shape, target.shape)
    mask_prediction = (prediction > 0.01) & (prediction < 0.98)
    mask_target = (target > 0.01) & (target < 0.98)
    mask = mask_prediction & mask_target

    
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

    return prediction_new

def compare_depth(depth_ALL, depth_mono, output_show_file):
    """
    比较depth_ALL和depth_mono的差异
    """
    depth_mono = np.array(depth_mono)/255.0
    depth_ALL = depth_ALL/255.0
    depth_mono = cv2.resize(np.array(depth_mono, dtype=np.float32), (depth_ALL.shape[1], depth_ALL.shape[0]))

    # print(depth_ALL.shape, depth_mono.shape)
    depth_ALL_ = depth_ALL[(depth_ALL > 0)&(depth_ALL < 0.99*depth_ALL.max())].reshape(-1, 1)
    depth_mono_ALL = depth_mono[(depth_ALL > 0)&(depth_ALL < 0.99*depth_ALL.max())].reshape(-1, 1)

    depth_mono_ALL = compute_scale_and_shift(depth_mono_ALL, depth_ALL_)

    # 展示depth_ALL和depth_mono的概率分布

    # 将两个维度的数据进行量化，然后生成曲线密度图
    d = np.vstack([depth_ALL_, depth_mono_ALL])
    xmin, xmax = depth_ALL_.min(), depth_ALL_.max()
    ymin, ymax = depth_mono_ALL.min(), depth_mono_ALL.max()
    rows, cols = 50, 50
    x, y = np.linspace(xmin, xmax, cols), np.linspace(ymin, ymax, rows)
    x, y = np.meshgrid(x, y)
    z = np.zeros_like(x)
    for i in range(rows):
        for j in range(cols):
            z[i, j] = np.sum((depth_ALL_ >= x[i, j]) & (depth_ALL_ < x[i, j] + (xmax-xmin)/cols) & (depth_mono_ALL >= y[i, j]) & (depth_mono_ALL < y[i, j] + (ymax-ymin)/rows))
    # 拟合线性方程

    # from sklearn.linear_model import (LinearRegression, HuberRegressor, RANSACRegressor, TheilSenRegressor)
    # import pandas as pd
    # lr = LinearRegression().fit(depth_ALL_, depth_mono_ALL)
    # coef_list = [["lr_coef", float(lr.coef_[0])]]
    # plotline_X = np.arange(depth_ALL_.min(), depth_ALL_.max(), 0.1).reshape(-1, 1)
    # fit_df = pd.DataFrame( index = plotline_X[:, 0], data={"linear_regression": lr.predict(plotline_X).flatten()})


    # ransac = RANSACRegressor(random_state=42).fit(depth_ALL_, depth_mono_ALL)
    # fit_df["ransac_regression"] = ransac.predict(plotline_X)
    # ransac_coef = ransac.estimator_.coef_
    # coef_list.append(["ransac_regression", ransac.estimator_.coef_[0]])
    # inlier_mask = ransac.inlier_mask_
    # outlier_mask = ~inlier_mask
    # print(f"Total outliers: {sum(outlier_mask)/len(depth_ALL_)}")


    # # theilsen = TheilSenRegressor(random_state=42).fit(depth_ALL_, depth_mono_ALL)
    # # fit_df["theilsen_regression"] = theilsen.predict(plotline_X)
    # # coef_list.append(["theilsen_regression", theilsen.coef_[0]])
    # # print(f"Outliers you added yourself: {sum(outlier_mask[:N_OUTLIERS])} / {N_OUTLIERS}")

    # huber = HuberRegressor().fit(depth_ALL_, depth_mono_ALL)
    # fit_df["huber_regression"] = huber.predict(plotline_X)
    # coef_list.append(["huber_regression", huber.coef_[0]])

    # fix, ax = plt.subplots()
    plt.contourf(x, y, z, 20, cmap='binary')
    # fit_df.plot(ax=ax, colormap='coolwarm')

    plt.colorbar()
    plt.xlabel('dense_nearlyGT')
    plt.ylabel('monodepth')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(output_show_file)
    plt.close()
    # plt.savefig("depth_map_compare.jpg")
    # print(coef_list)

if __name__ == "__main__":
    

    scenes = os.listdir(args.root_path)
    for scene_id in scenes:
        if args.root_path.endswith("iphone"):
            depth_gt_dir = os.path.join(args.root_path, scene_id, "depth_maps")
            depth_mono_dir = os.path.join(args.root_path, scene_id, "depth_maps_DPT")
        elif args.root_path.endswith("kinect"):
            depth_gt_dir = os.path.join(args.root_path, scene_id, "depth_maps2")
            depth_mono_dir = os.path.join(args.root_path, scene_id, "depth_maps_DPT")
        output_show_path = os.path.join(args.root_path, scene_id, "compare_gt_with_DPT")
        if not os.path.exists(output_show_path):
            os.makedirs(output_show_path)


        files = [i for i in os.listdir(depth_gt_dir) if (i.endswith(".jpg") or i.endswith(".png"))]
        if args.root_path.endswith("iphone"):
            images_train = [i.split("-")[0]+'_'+i.split('_')[-1] for i in files if (i.endswith("_train.jpg") or i.endswith("_train.png"))]
            images_test = [i.split("-")[0] +'.'+ i.split('.')[-1] for i in files if not (i.endswith("_train.jpg") or i.endswith("_train.png"))]
            images = images_train + images_test
        elif args.root_path.endswith("kinect"):
            images = [i.split('.')[0] for i in files]

        
        for idx, image in enumerate(images):
            depth_gt_file = os.path.join(depth_gt_dir,files[idx])
            depth_mono_file = os.path.join(depth_mono_dir, "depth_" + image.split('.')[0]+'.jpg')
            if not os.path.exists(depth_mono_file):
                depth_mono_file = os.path.join(depth_mono_dir, "depth_" + image.split('.')[0] + ".png")
            
            
            output_show_file = os.path.join(output_show_path, image)
            # if os.path.exists(output_show_file):
            #     continue

            depth_ALL = cv2.imread(depth_gt_file, cv2.IMREAD_GRAYSCALE)
            depth_mono = cv2.imread(depth_mono_file, cv2.IMREAD_GRAYSCALE)
            compare_depth(depth_ALL, depth_mono, output_show_file)
            print(f"compare {depth_gt_file} with {depth_mono_file} done!")
            # plt.savefig(os.path.join(output_show_path, i + ".jpg"))
            # plt.close()
        
        # depth_ALL = cv2.imread(depth_gt_file)
        # depth_mono = np.load(depth_mono_file)


        # compare_depth(depth_ALL, depth_mono)
