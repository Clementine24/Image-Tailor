# -*- coding = utf-8 -*-
# @Time : 2022/6/3 16:36
# @Author : fan
# @File:test_homography.py
# @Software: PyCharm

import cv2
import numpy as np
import os
from stitch import stitch
import matplotlib.pyplot as plt
import test_SIFT


def get_homography(H, W):
    """
    Generate a homography matrix for testing.
    :param H: the height of image
    :param W: the width of image
    :return: a perspective transform matrix
    """
    # 取要变换的图像的四角的点
    corner_point = np.array([[0, 0],
                             [0, H - 1],
                             [W - 1, 0],
                             [W - 1, H - 1]], dtype=np.float32)

    # 定义四角变换完后的位置，为了保持对比性，我们固定了这四个变换完后的点
    pers_point = np.array([[50, 50],
                           [-20, H - 70],
                           [W + 30, -20],
                           [W - 40, H + 80]], dtype=np.float32)

    # pers_point = np.array([[10, 5],
    #                        [-5, H - 15],
    #                        [W + 20, -20],
    #                        [W - 10, H + 10]], dtype=np.float32)
    # 使用cv2的包用4对对应点求解3*3投影变换矩阵
    true_homo = cv2.getPerspectiveTransform(corner_point, pers_point)

    return true_homo


if __name__ == '__main__':
    # 读入图像
    image_path = '../origin/images'
    test_left = cv2.imread(os.path.join(image_path, 'expt1.jpg'))
    # test_right = cv2.imread(os.path.join(image_path, 'test_right.jpg'))
    # 对右侧图像做一定的投影变换
    true_homo = get_homography(test_left.shape[0], test_left.shape[1])
    test_right_warp = cv2.warpPerspective(test_left, true_homo, test_left.shape[:2])

    test_left = cv2.resize(test_left, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    test_right_warp = cv2.resize(test_right_warp, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)

    # 将左侧图像和做过变换的右侧图像输入到待测试的缝合程序，返回求解出的单应性矩阵和缝合后的图像
    # 需要测试不同算法时，仅需修改以下部分
    """
    测试一：
    测试实现的配准缝合算法是否正确
    """
    # blending_mode = "linear_blending"
    # stitched_image, experi_homo = stitch(test_left, test_right_warp, blending_mode, test_mode=True)

    # 计算投影矩阵的差，以平方差函数作为损失函数
    # difference = np.linalg.norm(true_homo - experi_homo)
    #
    # print('True homography transform matrix :\n', true_homo)
    # print('Experience homography transform matrix :\n', experi_homo)
    # print('The difference between two homography matrix is :\n', difference)
    # plt.figure(0)
    # plt.title("stitch_img")
    # plt.imshow(stitched_image[:, :, ::-1].astype(int))
    # # plt.title("warped right image")
    # # plt.imshow(test_right_warp[:, :, ::-1].astype(int))
    # plt.show()

    """
    测试二：
    测试自己实现的SIFT算法和cv2中的SIFT算法
    """
    blending_mode = "linear_blending"
    stitched_image_cv, experi_homo_cv = test_SIFT.stitch(test_left, test_right_warp, blending_mode, test_mode=True, test_sift=False)
    stitched_image_my, experi_homo_my = test_SIFT.stitch(test_left, test_right_warp, blending_mode, test_mode=True, test_sift=True)
    difference_cv = np.linalg.norm(true_homo - np.linalg.inv(experi_homo_cv))
    difference_my = np.linalg.norm(true_homo - np.linalg.inv(experi_homo_my))
    difference_sift = np.linalg.norm(np.linalg.inv(experi_homo_my) - np.linalg.inv(experi_homo_cv))

    print('True homography transform matrix :\n', true_homo)
    print('Experience homography transform matrix using cv2 :\n', np.linalg.inv(experi_homo_cv))
    print('Experience homography transform matrix using my SIFT :\n', np.linalg.inv(experi_homo_my))
    print('The difference between true homography matrix and cv2 matrix is :\n', difference_cv)
    print('The difference between true homography matrix and my matrix is :\n', difference_my)
    print('The difference between cv2 homography matrix and my matrix is :\n', difference_sift)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].set_title("the origin picture")
    axes[0].set_axis_off()
    axes[0].imshow(test_left[:, :, ::-1].astype(int))
    axes[1].set_title("Transformed picture")
    axes[1].set_axis_off()
    axes[1].imshow(test_right_warp[:, :, ::-1].astype(int))
    axes[2].set_title("Opencv's Result")
    axes[2].set_axis_off()
    axes[2].imshow(stitched_image_cv[:, :, ::-1].astype(int))
    axes[3].set_title("ours result")
    axes[3].set_axis_off()
    axes[3].imshow(stitched_image_my[:, :, ::-1].astype(int))
    plt.show()



