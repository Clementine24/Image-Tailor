import matplotlib.pyplot as plt
import cv2
import os
import time
from stitch import stitch
from stitch import *

if __name__ == "__main__":

    # 导入要拼接的两张图像
    image_path = "../origin/images"
    image_name_1 = 'exp2.png'
    image_name_2 = 'exp1.png'
    start = time.time()
    image_left = cv2.imread(os.path.join(image_path, image_name_1))
    image_left = cv2.resize(image_left, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    image_right = cv2.imread(os.path.join(image_path, image_name_2))
    image_right = cv2.resize(image_right, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    # 选择合适的参数对两张图片进行缝合
    blending_mode = "linear_blending"
    stitched_image_right = stitch(image_left, image_right, blending_mode)
    stitched_image_wrong = stitch(image_right, image_left, blending_mode)
    end = time.time()

    fig, axes = plt.subplots(1, 4, figsize=(12, 6))
    axes[0].imshow(image_left[:, :, ::-1].astype(int))
    axes[0].set_axis_off()
    axes[0].set_title('The left pic we process')
    axes[1].imshow(image_right[:, :, ::-1].astype(int))
    axes[1].set_axis_off()
    axes[1].set_title('The right pic we produce')
    axes[2].imshow(stitched_image_right[:, :, ::-1].astype(int))
    axes[2].set_axis_off()
    axes[2].set_title('The right order we produce')
    axes[3].imshow(stitched_image_wrong[:, :, ::-1].astype(int))
    axes[3].set_axis_off()
    axes[3].set_title('The wrong order we produce')

    plt.show()


