import copy

import numpy as np
import matplotlib.pyplot as plt

from sift_main import *


# First, we want to visualize the pyramid one by one
def visualize_pyramid(image: np.array, interval=3):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32')
    gaussian, dog = scale_space(img, num_intervals=interval)
    # for gaussian visualize
    fig, axes = plt.subplots(gaussian.__len__(), interval + 3,
                             figsize=(12, 12))
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].imshow(gaussian[i][j, :, :], cmap='gray')
            axes[i, j].set_axis_off()
    plt.savefig('Gaussian_Pyramid.jpeg')
    plt.show()

    fig, axes = plt.subplots(gaussian.__len__(), interval + 2,
                             figsize=(10, 12))
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].imshow(dog[i][j, :, :], cmap='gray')
            axes[i, j].set_axis_off()
    plt.savefig('DOG_Pyramid.jpeg')
    plt.show()


def features_extract(image: np.array):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32')
    # keypoints, _ = computeKeypointsAndDescriptors(img)
    gaussian, dog = scale_space(img)
    keypoints = detect_features(gaussian, dog)
    image_res = copy.deepcopy(image)

    for keypoint in keypoints:
        position = (int(keypoint.pt[0]), int(keypoint.pt[1]))
        cv2.circle(image_res, position, 3, (0, 0, 255), 1)

    sift = cv2.SIFT_create()
    key_points, _ = sift.detectAndCompute(image, None)
    image_sift = copy.deepcopy(image)
    for keypoint in key_points:
        position = (int(keypoint.pt[0]), int(keypoint.pt[1]))
        cv2.circle(image_sift, position, 3, (0, 0, 255), 1)

    fig, axes = plt.subplots(2, 1, figsize=(6, 12))
    axes[0].set_title(f"ours with matching points {keypoints.__len__()}")
    axes[0].imshow(image_res[:, :, ::-1].astype(int))
    axes[0].set_axis_off()
    axes[1].set_title(f"cv2 with matching points {key_points.__len__()}")
    axes[1].imshow(image_sift[:, :, ::-1].astype(int))
    axes[1].set_axis_off()
    plt.savefig('cptocv2.png')
    plt.show()


if __name__ == '__main__':

    # visualize_pyramid(exp)
    # features_extract(exp)
    # _, a = sift_detect(exp)
    import time
    import time

    exp = cv2.imread('exp2.png')
    exp = cv2.resize(exp, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    features_extract(exp)
    # start = time.time()
    # sift = cv2.SIFT_create()
    # sift.detectAndCompute(exp, None)
    # end = time.time()
    # print(exp.shape, "cv2", 'cost: ', end - start)
    # start = time.time()
    # sift_detect(exp)
    # end = time.time()
    # print(exp.shape, "ours", 'cost: ', end - start)
