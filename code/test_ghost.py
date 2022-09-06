import matplotlib.pyplot as plt
import cv2
import os
import time
from stitch import stitch


if __name__ == "__main__":

    # 导入要拼接的两张图像
    image_path = "../origin/images"
    image_name_1 = 'exps1.jpg'
    image_name_2 = 'exps2.jpg'
    image_left = cv2.imread(os.path.join(image_path, image_name_1))
    image_left = cv2.resize(image_left, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    image_right = cv2.imread(os.path.join(image_path, image_name_2))
    image_right = cv2.resize(image_right, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)

    # 选择合适的参数对两张图片进行缝合
    blending_mode = "linear_blending"
    stitched_image_linear = stitch(image_left, image_right, "linear_blending")
    stitched_image_no = stitch(image_left, image_right, "no_blending")

    # 展示缝合后的图片
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title("the result with no blending")
    axes[0].set_axis_off()
    axes[0].imshow(stitched_image_no[:, :, ::-1].astype(int))
    axes[1].set_title("the result with linear blending")
    axes[1].set_axis_off()
    axes[1].imshow(stitched_image_linear[:, :, ::-1].astype(int))
    plt.show()
