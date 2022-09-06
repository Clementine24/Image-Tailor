import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import sys
sys.path.append("..")
from SIFT.sift_main import sift_detect


from utils import *


def get_key_points_and_features(image):
	"""
	使用SIFT算法检测输入图像的关键点和对应特征
	输入：
		image：3维numpy矩阵，待处理图像
	输出：
		key_points: 图像检测出的关键点
		features: 每一个关键点对应的特征向量
	"""
	sift = cv2.SIFT_create()
	key_points, features = sift.detectAndCompute(image, None)
	# return sift_detect(image)
	return key_points, features


def match_key_points(key_points_left, key_points_right, features_left, features_right, ratio_threshold):
	"""
	一一匹配两张图片的特征点
	输入：
		key_points_left：左边图片的关键点
		key_points_right：右边图片的关键点
		features_left：左边图片关键点对应的特征向量
		features_right：右边图片关键点对应的特征向量
		ratio_threshold：特征匹配度阈值
	输出：
		matched_positions：元素为元组的列表，左右图对应特征点的坐标
	"""
	# 对左图每一个特征点，遍历右图的特征点，并找到第一和第二匹配的特征点，作为候选特征对
	candidate_pairs = []
	for i, feature_left in enumerate(features_left):
		# 初始化
		min_index = -1
		min_distance = np.inf
		second_min_distance = np.inf

		# 遍历右图每一个特征点
		for j, feature_right in enumerate(features_right):
			cur_distance = np.linalg.norm(feature_left - feature_right) # 使用二范数度量特征向量的差
			# 更新第一匹配特征点
			if cur_distance < min_distance:
				min_index = j
				min_distance = cur_distance
			# 更新第二匹配特征点
			elif cur_distance < second_min_distance != min_distance:
				second_min_distance = cur_distance
		candidate_pairs.append((min_index, min_distance, second_min_distance))

	# 判断第一匹配特征对是否足够显著
	matched_indices = []
	for i, (min_index, min_distance, second_min_distance) in enumerate(candidate_pairs):
		if min_distance / second_min_distance <= ratio_threshold:
			matched_indices.append((i, min_index))

	# 取出左右两张图对应特征点的坐标
	matched_positions = []
	for left_index, right_index in matched_indices:
		position_left = (int(key_points_left[left_index].pt[0]), int(key_points_left[left_index].pt[1]))
		position_right = (int(key_points_right[right_index].pt[0]), int(key_points_right[right_index].pt[1]))
		matched_positions.append((position_left, position_right))

	return matched_positions


def fit_perspective_matrix(matched_positions, num_iterations=5000, inlier_threshold=5, num_samples=4, measurement='l2', txt=None):
	"""
	使用RANSAC算法计算变换的单应性矩阵
	输入：
		matched_positions：列表，对应的特征点的坐标
		num_iterations：最大循环次数
		inlier_threshold：inlier最小阈值
		num_samples：每次随机采样特征对数量
		measurement：字符串，度量特征向量差值的方法
	输出：
		best_H：最佳的投影变换矩阵
	"""
	assert measurement in {'l1', 'inf', 'l2'}

	# 提取对应的左右特征点
	left_points, right_points = zip(*matched_positions)
	left_points = np.array(left_points)
	right_points = np.array(right_points)

	# 应用RANSAC算法，计算单应性矩阵
	# 设置超参数
	num_position_pairs = len(matched_positions) # 总特征对数量
	num_max_inliers = -1  # 最多inlier
	best_H = None  # 最佳单应性矩阵

	# 如果采样点取值为none，则默认为使用所有特征对计算单应性矩阵
	if num_samples is None:
		num_iterations = 1
		num_samples = len(matched_positions)

	# 度量特征向量距离的方法
	if measurement == "l1":
		norm_ord = 1
	elif measurement == "inf":
		norm_ord = np.inf
	else:
		norm_ord = 2

	# 循环应用RANSAC算法
	for _ in range(num_iterations):
		# 随机从左右特征对中抽取一定数量特征对，并计算单应性矩阵
		sampled_indices = random.sample(range(num_position_pairs), num_samples)
		H = solve_perspective_matrix(right_points[sampled_indices], left_points[sampled_indices])

		# 计算该循环的单应性矩阵性能，用inlier的数量评价
		num_inliers = 0
		for i in range(num_position_pairs):
			right_point = right_points[i]
			left_point = left_points[i]
			if i not in sampled_indices:
				mapped_left_point = H @ np.hstack((right_point, 1))
				if mapped_left_point[2] <= 1e-8:   # 避免计算溢出
					continue

				# 计算右图特征点投影变换后对应左图的坐标
				mapped_left_point /= mapped_left_point[2]
				# 如果坐标之差小于给定阈值，则判定其为inlier
				if np.linalg.norm(mapped_left_point[:2] - left_point, norm_ord) < inlier_threshold:
					num_inliers += 1

		# 更新最佳单应性矩阵和最大inlier数量
		if num_max_inliers < num_inliers:
			num_max_inliers = num_inliers
			best_H = H

	print(f"The number of inliers of the best homograph matrix: {num_max_inliers}")
	if txt:
		txt.insert('end', f"The number of inliers of the best homograph matrix: {num_max_inliers}\n")

	return best_H


def warp(image_left, image_right, H, blending_mode):
	"""
	变换右图图像矩阵并进行拼接
	输入：
		image_left：三维numpy矩阵，左图
		image_right：三维numpy矩阵，原始右图
		H：numpy矩阵，单应性矩阵
		blending_mode：左右两图混合模式，有三种可选模式：no_blending、linear_blending
	输出：
		stitch_img：拟合后的图像
	"""
	assert blending_mode in {'no_blending', 'linear_blending'}

	# 初始化输出的缝合图像画布
	height_left, width_left, _ = image_left.shape
	height_right, width_right, _ = image_right.shape
	height_stitch = max(height_left, height_right)
	width_stitch = width_left + width_right
	stitch_img = np.zeros((height_stitch, width_stitch, 3), dtype=np.uint8)

	# 不对两图进行拟合，则直接将右图覆盖在左图上
	if blending_mode == "no_blending":
		stitch_img[:height_left, :width_left] = image_left

	# 对输出图像各个像素点进行反向变换，如果坐标落在右图范围内，则对右图像素点进行采样
	inv_H = np.linalg.inv(H)
	for h in range(height_stitch):
		for w in range(width_stitch):
			# 反向变换
			original_left_point = np.array([w, h, 1])
			mapped_right_point = inv_H @ original_left_point
			mapped_right_point /= mapped_right_point[2]
			
			y, x, _ = mapped_right_point
			# 判断反向变换后的像素点是否在右图范围内
			if 0 < x < height_right and 0 < y < width_right:
				# 若在右图范围内，则进行双线性插值，否则跳过该点
				stitch_img[h, w] = bilinear_interpolation_single(image_right, (x, y))

	# 使用线性拟合的方法缝合两图
	if blending_mode == "linear_blending":
		stitch_img = linear_blending(image_left, stitch_img)

	# 除去画布四周多余的黑边
	stitch_img = removeBlackBorder(stitch_img)

	return stitch_img


def stitch(image_left, image_right, blending_mode="linear_blending", ratio_threshold=0.75, test_mode=False, txt=None):
	"""
	缝合图像的主函数
	输入：
		image_left：三维numpy矩阵，待缝合的左图
		image_right：三维numpy矩阵，待缝合的右图
		blending_mode：字符串，图像拟合模式
		ratio_threshold：关键特征对选取阈值
		test_mode：bool，测试模式。如果为测试模式，则同时输出中间计算的单应性矩阵结果
	输出：
		warped_image：裁剪后的图像
		H：投影变换的单应性矩阵
	"""
	# step1：使用SIFT算法查找两图的关键点和对应的特征向量
	print("Step1 - Use SIFT algorithm extract the key points and features...")
	if txt:
		txt.insert('end', "Step1 - Use SIFT algorithm extract the key points and features...\n")
	key_points_left, features_left = get_key_points_and_features(image_left)
	key_points_right, features_right = get_key_points_and_features(image_right)

	# step2：匹配两图的对应特征对，并选出足够显著的特征对
	print("Step2 - Match the key points of two images...")
	if txt:
		txt.insert('end', "Step2 - Match the key points of two images...\n")
	matched_positions = match_key_points(key_points_left, key_points_right, features_left, features_right, ratio_threshold)
	print("The number of matching points:", len(matched_positions))

	# 可视化带有关键点对的图像
	matching_image_origin, matching_image = draw_matched_points(image_left, image_right, matched_positions)

	# Step3：使用RANSAC算法循环计算最佳的单应性投影矩阵
	print("Step3 - Find the best homography matrix with RANSAC algorithm...")
	if txt:
		txt.insert('end', "Step3 - Find the best homography matrix with RANSAC algorithm...\n")
	H = fit_perspective_matrix(matched_positions, txt=txt)

	# Step4：对图像进行投影变换并缝合
	print("Step4 - Warp and stitch images...")
	if txt:
		txt.insert('end', "Step4 - Warp and stitch images...\n")
	warped_image = warp(image_left, image_right, H, blending_mode)

	if test_mode:
		return matching_image_origin, matching_image, warped_image, H
	else:
		return matching_image_origin, matching_image, warped_image
