import matplotlib.pyplot as plt
import cv2
import os
import time
from stitch import *
from test_homography import *

if __name__ == "__main__":
    # 导入要拼接的两张图像
    image_path = "../origin/images"
    image_name_1 = 'expt1.jpg'
    # 先旋转变换
    image_left = cv2.imread(os.path.join(image_path, image_name_1))
    true_homo = get_homography(image_left.shape[0], image_left.shape[1])
    image_right = cv2.warpPerspective(image_left, true_homo, image_left.shape[:2])
    # 然后缩放，保持特征
    image_left = cv2.resize(image_left, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    image_right = cv2.resize(image_right, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    true_homo = get_homography(image_right.shape[0], image_right.shape[1])
    image_right = cv2.warpPerspective(image_right, true_homo, image_right.shape[:2])
    print("Step1 - Use SIFT algorithm extract the key points and features...")
    key_points_left, features_left = get_key_points_and_features(image_left)
    key_points_right, features_right = get_key_points_and_features(image_right)

    # step2：匹配两图的对应特征对，并选出足够显著的特征对
    print("Step2 - Match the key points of two images...")
    matched_positions = match_key_points(key_points_left, key_points_right, features_left, features_right, 0.75)
    print("The number of matching points:", len(matched_positions))

    # Step3：使用RANSAC算法循环计算最佳的单应性投影矩阵
    print("Step3 - Find the best homography matrix with RANSAC algorithm...")
    num = [4, 8, 16, 24, 36, 64]
    meas = ['l2', 'l1', 'inf']
    H_l2 = []
    H_l1 = []
    H_inf = []
    for i in num:
        H_l2.append(fit_perspective_matrix(matched_positions, num_samples=i, measurement='l2'))
        H_l1.append(fit_perspective_matrix(matched_positions, num_samples=i, measurement='l1'))
        H_inf.append(fit_perspective_matrix(matched_positions, num_samples=i, measurement='inf'))
    H = [H_l2, H_l1, H_inf]

    # Step4：对图像进行投影变换并缝合
    blending_mode = 'no_blending'
    print("Step4 - Warp and stitch images...")
    fig, axes = plt.subplots(3, len(num), figsize=(18, 9))
    for i in range(len(num)):
        for j in range(3):
            axes[j, i].imshow(warp(image_left, image_right, H[j][i], blending_mode)[:, :, ::-1].astype(int))
            axes[j, i].set_axis_off()
            axes[j, i].set_title(f"{num[i]} points with {meas[j]}")
            print(np.linalg.norm(np.linalg.inv(H[j][i]) - true_homo))
