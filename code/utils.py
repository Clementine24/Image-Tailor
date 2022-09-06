import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


def removeBlackBorder(image):
	"""
	移除缝合后的图像的多余黑边
	输入：
		image：三维numpy矩阵，待处理图像
	输出：
		裁剪后的图像
	"""
	height, width, _ = image.shape
	reduced_height, reduced_width = height, width

	# 鉴于我们的程序拼接图像是从左向右拼接，因此这里使用从右到左查找黑边边界范围
	for col in range(width - 1, -1, -1):
		for i in range(height):
			if np.count_nonzero(image[i, col]) > 0:
				break
		else:
			reduced_width = reduced_width - 1
	
	# 拼接图像是从上到下拼接，因此这里使用从下到上查找黑边边界范围
	for row in range(height - 1, -1, -1):
		for i in range(reduced_width):
			if np.count_nonzero(image[row, i]) > 0:
				break
		else:
			reduced_height = reduced_height - 1
	
	return image[:reduced_height, :reduced_width]


def solve_perspective_matrix(right_points, left_points):
	"""
	求解单应性矩阵
	输入：
		right_points：列表，左图对应的特征点
		left_points：列表，右图对应的特征点
	输出：
		H：三维列表，计算好的单应性矩阵
	:param right_points: coordinates of the points in the original plane
	:param left_points: coordinates of the points in the target plane
	:return H: the perspective matrix
	"""
	A = []
	for r in range(len(right_points)):
		A.append([
			-right_points[r, 0], -right_points[r, 1], -1, 0, 0, 0,
			right_points[r, 0] * left_points[r, 0], right_points[r, 1] * left_points[r, 0], left_points[r, 0]
		])
		A.append([
			0, 0, 0, -right_points[r, 0], -right_points[r, 1], -1,
			right_points[r, 0] * left_points[r, 1], right_points[r, 1] * left_points[r, 1], left_points[r, 1]
		])

	# 使用SVD算法求解等式Ah=0
	u, s, vh = np.linalg.svd(A)
	# 从vh中取出单应性矩阵（具体数学推导见报告）
	H = vh[8].reshape((3, 3))
	# 对单应性矩阵归一化
	H /= H.item(8)
	
	return H


def linear_blending(image_left, image_right):
	"""
	实现线性拟合算法
	输入：
		image_left：三维numpy矩阵，待拟合的左图
		image_right：三维numpy矩阵，待拟合的右图
	输出：
		blended_image：三维numpy矩阵，拟合后的图片
	"""
	height_left, width_left, _ = image_left.shape
	height_right, width_right, _ = image_right.shape
	image_left_mask = np.zeros((height_right, width_right), dtype=bool)
	image_right_mask = np.zeros((height_right, width_right), dtype=bool)

	# 分别寻找左右图的范围，即灰度值不为零的像素点位置
	for i in range(height_left):
		for j in range(width_left):
			if np.count_nonzero(image_left[i, j]) > 0:
				image_left_mask[i, j] = True
	for i in range(height_right):
		for j in range(width_right):
			if np.count_nonzero(image_right[i, j]) > 0:
				image_right_mask[i, j] = True
	
	# 查找两图重叠的区域，创建重叠的mask
	overlap_mask = np.zeros((height_right, width_right), dtype=bool)
	for i in range(height_right):
		for j in range(width_right):
			if image_left_mask[i, j] and image_right_mask[i, j]:
				overlap_mask[i, j] = True
	
	# 根据像素点距离左右两图的距离，计算重叠区域线性拟合的比例
	blending_mask = np.zeros((height_right, width_right))
	for i in range(height_right):
		min_index = max_index = -1
		for j in range(width_right):
			# 判断重叠区域左边界
			if overlap_mask[i, j] and min_index == -1:
				min_index = j
			# 更新重叠区域右边界
			if overlap_mask[i, j]:
				max_index = j
		
		if min_index == max_index:  # 说明这一横行没有重叠区域，无需拟合
			continue
		
		# 根据插值方法计算线性拟合的比例，简单来说，距离左图越近，左图占比越大；距离右图越近，右图占比越大
		decrease_step = 1 / (max_index - min_index)
		for j in range(min_index, max_index + 1):
			blending_mask[i, j] = 1 - (decrease_step * (j - min_index))
	
	# 创建输出的线性拟合画布
	blended_image = np.copy(image_right)
	blended_image[:height_left, :width_left] = np.copy(image_left)
	
	# 对每个像素点，判断是否在重叠区域，如果是，则需要应用线性拟合
	for i in range(height_right):
		for j in range(width_right):
			if overlap_mask[i, j]:
				blended_image[i, j] = blending_mask[i, j] * image_left[i, j] + (1 - blending_mask[i, j]) * image_right[i, j]
	
	return blended_image


def draw_matched_points(image_left, image_right, matched_positions):
	"""
	可视化两张图对应的关键点
	输入：
		image_left：三维numpy矩阵，左图
		image_right：三维numpy矩阵，右图
		matched_position：列表，对应的关键点
	输出：
		matching_image_origin：三维numpy矩阵，两图像并排放置的结果
		matching_image：三维numpy矩阵，在并排放置的基础上，加了特征点和对应特征连线的图像
	"""

	# 初始化输出画布，并将左右两图添加到画布上
	height_left, width_left, _ = image_left.shape
	height_right, width_right, _ = image_right.shape
	matching_image = np.zeros((max(height_left, height_right), width_left + width_right, 3), dtype=np.uint8)
	matching_image[:height_left, :width_left] = image_left
	matching_image[:height_right, width_left:] = image_right
	matching_image_origin = np.copy(matching_image)

	# 在画布上添加特征点和对应连线
	for left_position, right_position in matched_positions:
		center_left = left_position
		center_right = (right_position[0] + width_left, right_position[1])
		cv2.circle(matching_image, center_left, 3, (153, 0, 102), 1)
		cv2.circle(matching_image, center_right, 3, (0, 102, 51), 1)
		cv2.line(matching_image, center_left, center_right, (255, 255, 153), 1)

	return matching_image_origin, matching_image


def bilinear_interpolation_single(image, coord):
	"""
	对单一坐标进行双线性插值
	输入：
		image: 三维numpy矩阵，要在其上进行插值
		coord: 两个元素的元组，代表坐标（浮点数）
	输出：
		intensity：三个元素的numpy数组，代表该坐标三个通道的强度值
	"""
	height, width, _ = image.shape
	# 浮点坐标
	x, y = coord
	# 四个邻居点坐标
	x1, y1 = math.floor(x), math.floor(y)
	x2, y2 = min(x1 + 1, height - 1), min(y1 + 1, width - 1)
	# 线性权重
	s, r = x - x1, y - y1
	# 四个邻居点的强度
	p0, p1, p2, p3 = image[x1, y1], image[x1, y2], image[x2, y1], image[x2, y2]
	# 双线性加权得到浮点坐标的强度值
	intensity = np.round((1-s)*(1-r) * p0 + (1-s)*r * p1 + s*(1-r) * p2 + s*r * p3)
	return intensity
