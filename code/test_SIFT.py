# -*- coding = utf-8 -*-
# @Time : 2022/6/4 11:40
# @Author : fan
# @File:test_SIFT.py
# @Software: PyCharm

from stitch import match_key_points, fit_perspective_matrix, warp
from SIFT.sift_main import sift_detect
import cv2


def get_key_points_and_features_cv(image):
    """
    检测给定图片的关键点和对应特征
    """
    sift = cv2.SIFT_create()
    key_points, features = sift.detectAndCompute(image, None)
    return key_points, features


def get_key_points_and_features_my(image):
    """
    使用自己完成的sift算法检测关键点和对应的特征
    """
    return sift_detect(image)


def stitch(image_left, image_right, blending_mode="linear_blending", ratio_threshold=0.75, test_mode=False,
           test_sift=True):
    """
    缝合图像的主函数，添加参数test_sift，来区分是否使用自己完成的sift算法
    """
    h_left, w_left, _ = image_left.shape
    h_right, w_right, _ = image_right.shape

    print("Step1 - Extract the key points and features by SIFT detector and descriptor...")
    if test_sift:
        key_points_left, features_left = get_key_points_and_features_my(image_left)
        key_points_right, features_right = get_key_points_and_features_my(image_right)
    else:
        key_points_left, features_left = get_key_points_and_features_cv(image_left)
        key_points_right, features_right = get_key_points_and_features_cv(image_right)

    print("Step2 - Extract the match point with threshold (David Lowe’s ratio test)...")
    matched_positions = match_key_points(key_points_left, key_points_right, features_left, features_right,
                                         ratio_threshold)
    print("The number of matching points:", len(matched_positions))

    # Step3 - fit the homography model with RANSAC algorithm
    print("Step3 - Fit the best homography model with RANSAC algorithm...")
    P = fit_perspective_matrix(matched_positions)

    # Step4 - Warp image to create panoramic image
    print("Step4 - Warp image to create panoramic image...")
    warped_image = warp(image_left, image_right, P, blending_mode)

    if test_mode:
        return warped_image, P
    else:
        return warped_image
