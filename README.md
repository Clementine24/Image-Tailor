# Image Tailor
该项目是我在复旦大学DATA130049课程上的合作期末project。利用图像配准技术拼接两图，使用到SIFT算法、SANSAC算法等。详细内容见report_final.pdf。

## 开发环境
基于python3.8 开发，需要安装如下package。由于是在windows下开发的，在linux系统上可能遇到未知问题。如有发现，可以及时联系作者
```bash
# 安装cv2库
pip install opencv-python
# 安装PIL库
pip install pillow
# 安装numpy包
pip install numpy
```

## Software Architecture

* `SIFT/`: 基于python实现的sift算法。
  * `sift_detect`: 提取sift关键点与特征向量。 
  * `visualize`: 一些可视化接口
* `images/`: 所有实验中用到的实验图片和输出结果。
  * `Origin/`: 原始图像部分
  * `Process/`: 输出图像部分
* `code/`: 图像拼接代码
  * 源码部分：
    * `stitch.py`: 图像拼接部分代码
    * `visual.py`: 可视化GUI源码
    * `utiles.py`: 实用接口工具，用于输出图片结果
  * 实验部分：
    * `test_blending.py`: 用于测试不同blending模式下的不同效果
    * `test_homography.py`: 自主实现的SIFT与Opencv对比 
    * `test_SIFT.py`: 单应性矩阵实验源码
    * `test_ghost.py`: '鬼影'实验源代码
    * `test_measurement.py`: 测试不同范数条件、不同拟合取点下的拟合效果
    * `test_order`: 测试不同的图片摆放顺序下的输出结果
