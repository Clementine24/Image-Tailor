import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)
float_tolerance = 1e-7
MAX_OCTAVES = 8


def sift_detect(img: np.array, sigma=1.6, num_intervals=3,
                assumed_blur=0.5, image_border_width=5):
    """计算 SIFT keypoints（特征点） and desccriptions（特征向量）

    输入：
        :param img: np.array, 输入的图像
        :param sigma: the blur parameter，作为初始化的sigma函数
        :param num_intervals: 在每一个octaves上进行提取的特征层数。
        :param assumed_blur: 相机通常会对图片进行一次高斯模糊，所以图像处理的时候需要考虑
        :param image_border_width: 内点极值检测需要忽略的边界

    输出：
        :return: keypoints: 作为cv2.keypoints对象，存储keypoints的基本信息。
        :return: desccriptions: 特征向量，用于后续的特征
    """
    # Step 0: 读取信息并且转换为二值图，并且对于灰度值进行归一化，利于后续梯度计算。
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = image.astype('float32')

    # PART 1: 构建 Scale Space
    gaussian_images, dog_images = scale_space(image, sigma, num_intervals, assumed_blur)

    # PART 2: 提取特征
    keypoints = detect_features(gaussian_images, dog_images,
                                      num_intervals, sigma, image_border_width)

    # Part3: 给出描述特征向量
    descriptor = descriptors(keypoints, gaussian_images)
    return keypoints, descriptor


def scale_space(image, sigma=1.6, num_intervals=3, assumed_blur=0.5):
    """高斯金字塔与高斯差分金字塔的创建。

    输入：
        :param img: np.array, 输入的图像
        :param sigma: the blur parameter，作为初始化的sigma函数
        :param num_intervals: 在每一个octaves上进行提取的特征层数。
        :param assumed_blur: 相机通常会对图片进行一次高斯模糊，所以图像处理的时候需要考虑

    输出：
        :return: gaussian_images: 高斯金字塔
        :return: dog_images: 高斯差分金字塔
    """
    # 构建 Image Pyramids for Scale Space.
    # Step 1: 这边对于图片先进行二倍扩充。这样可以获得对于图片更加详细的信息。
    image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    # change the sigma_diff, since the original pic has the assumed_blur
    sigma_diff = np.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    base_image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)

    # Step 2: 构建金字塔。计算 how may octaves 我们需要计算。由于我们先扩大了图片二倍, so minus 1
    num_octaves = min(round(np.log2(min(base_image.shape)) - 1), 8)  # following cv2

    # 递归地构造高斯核函数。关键点在于这里可以一层一层对齐使用高斯滤波，进行卷积，并减少运算量。具体内容可以参见报告
    # sigma_total 是对于原图进行的高斯滤波总额，而gaussian_kernels[image_index]计算的是对于上一步
    # 生成的图片需要进行多少高斯滤波达到 sigma_total 的效果。
    sigma_total = np.asarray(
        [sigma * (2 ** (index / float(num_intervals))) for index in range(0, num_intervals + 3)],
        dtype=np.float64
    )
    gaussian_kernels = np.sqrt(sigma_total[1:] ** 2 - sigma_total[0:-1] ** 2)

    gaussian_images = []
    dog_images = []
    image_dealt = base_image
    # 造金字塔流程（包括高斯金字塔，与DOG金字塔，一起运算减少时间开销）
    for _ in range(num_octaves):
        octave_layers = [image_dealt]
        dog_layers = []
        # 对于每一层的图像，连续进行高斯模糊操作。这里好处在于，递归操作可以减小卷积核，进而提高算法速度。
        for kernel in gaussian_kernels:
            image_dealt = cv2.GaussianBlur(image_dealt, (0, 0), sigmaX=kernel, sigmaY=kernel)
            octave_layers.append(image_dealt)  # 这一层的layer
            dog_layers.append(np.subtract(octave_layers[-1], octave_layers[-2]))  # 这层减去上一层的
        gaussian_images.append(np.asarray(octave_layers))  # 这是每一层的高斯模糊的结果
        dog_images.append(np.asarray(dog_layers))  # 这是每一层的高斯模糊的结果
        octave_base = octave_layers[-3]  # 更新下一层的初始图片，降采样操作，使用的正好是倒数第三章图片
        image_dealt = cv2.resize(octave_base, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

    return gaussian_images, dog_images


def detect_features(gaussian_images, dog_images, num_intervals=3, sigma=1.6, border_width=5,
                    contrast_threshold=0.04):
    """寻找极值点所在位置和精确坐标，完成主方向确定工作

    输入：
        :param gaussian_images: 高斯金字塔， 用于计算极值点的主方向
        :param dog_images: 高斯差分金字塔, 用于定位极值点
        :param sigma: the blur parameter，作为初始化的sigma函数
        :param num_intervals: 在每一个octaves上进行提取的特征层数。
        :param border_width: 内点极值检测需要忽略的边界
        :param contrast_threshold: 超参数，用于threshold过滤不显著的特征点

    输出：
        :return: keypoints: 特征点向量
    """
    threshold = contrast_threshold / num_intervals * 255  # from OpenCV implementation
    keypoints = []

    def is_extremum(subregion, thre):
        """ 判断是否是极值点
        """
        center_pixel_value = subregion[1, 1, 1]
        if abs(center_pixel_value) > thre:  # 对于不显著的点进行过滤
            if center_pixel_value > 0:
                return 13 == np.argmax(subregion)  # 最大点是否在中心？（中心是13）
            elif center_pixel_value < 0:
                return 13 == np.argmin(subregion)  # 最大点是否在中心？（中心是13）
        return False

    for octave_index, dog_layers in enumerate(dog_images):
        _, x_size, y_size = dog_layers.shape
        # 首先，按照顺序，寻找极值点
        for image_index in range(0, num_intervals):
            for i in range(border_width, x_size - border_width):
                for j in range(border_width, y_size - border_width):
                    if is_extremum(dog_layers[image_index:image_index + 3, i - 1:i + 2, j - 1:j + 2], threshold):
                        # 如果是极值点，找到精确的极值点所在位置，使用迭代方式寻找
                        localization_result = exact_features(i, j, image_index + 1, octave_index, num_intervals,
                                                             dog_layers, sigma, contrast_threshold, border_width)
                        if localization_result is not None:
                            # print(f"detect keypoint at {i}, {j}, {image_index+1}, {octave_index}")
                            keypoint, localized_image_index = localization_result
                            # 对于特征点，寻找其主方向。
                            keypoints_with_orientations = com_Orientations(keypoint, octave_index,
                                                                           gaussian_images[octave_index][
                                                                           localized_image_index, :, :])
                            for keypoint_with_orientation in keypoints_with_orientations:
                                keypoints.append(keypoint_with_orientation)

    # 由于同一块地方可能有重复特征，需要进行筛选。下面定义的是equal 函数。
    def keypoints_eq(keypoint1, keypoint2):
        return keypoint1.pt[0] == keypoint2.pt[0] and \
               keypoint1.pt[1] == keypoint2.pt[1] and \
               keypoint1.size == keypoint2.size and \
               keypoint1.angle == keypoint2.angle

    unique_keypoints = []     # 去除重复的特征。。。
    if len(keypoints) <= 1:
        unique_keypoints = keypoints
    else:
        for cur_keypoint in keypoints:
            for pre_keypoint in unique_keypoints:
                if keypoints_eq(cur_keypoint, pre_keypoint):
                    break
            else:
                unique_keypoints.append(cur_keypoint)

    # 将坐标恢复到原来的图片上
    for keypoint in unique_keypoints:
        keypoint.pt = (0.5 * keypoint.pt[0], 0.5 * keypoint.pt[1])  # 我们把图片放大了两倍，不要忘记哦。。。
        keypoint.size *= 0.5  # 同样的，放大倍数需要缩小 1/2
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)  # ??? 直接看opencv的。。。

    return unique_keypoints


def exact_features(i, j, image_index, octave_index, num_intervals, dog_layers,
                   sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10,
                   max_iter=5):
    """精确定位极值点

    输入：
        :param i: 粗略确定的极值点横坐标
        :param j: 粗略确定的极值点纵坐标
        :param image_index: 粗略确定的极值点所在octaves的第image_index层
        :param octave_index: 图片所在的金字塔层数
        :param num_intervals: 在每一个octaves上进行提取的特征层数。
        :param dog_layers: 所在的高斯差分金字塔，用于特征迭代过程
        :param sigma: 方差系数，用于确定放大倍数
        :param contrast_threshold: 超参数，用于控制特征数量。
        :param image_border_width: 内点需要忽略的边界数量
        :param eigenvalue_ratio: 特征值，用于图像边缘的特征筛选。具体在代码中会提到
        :param max_iter: 迭代的最大次数

    输出：
        :return: keypoints: 筛选出的精确特征点
    """

    def cal_Grad(pixel_array):
        """计算梯度
        """
        dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
        dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
        ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
        return np.array([dx, dy, ds])

    def cal_Hessian(pixel_array):
        """计算海森矩阵
        """
        center_pixel_value = pixel_array[1, 1, 1]
        dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
        dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
        dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
        dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
        dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
        dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
        return np.array([[dxx, dxy, dxs],
                         [dxy, dyy, dys],
                         [dxs, dys, dss]])

    # 初值设定
    image_shape = dog_layers[0].shape
    pixel_cube = dog_layers[image_index - 1:image_index + 2, i - 1:i + 2, j - 1:j + 2].astype('float32') / 255
    gradient = cal_Grad(pixel_cube)
    hessian = cal_Hessian(pixel_cube)
    extremum_update = - np.linalg.lstsq(hessian, gradient, rcond=None)[0]
    # 五次迭代，找到可能的最小值（最小值可能不在那个点）
    for attempt_index in range(max_iter):
        # 结束条件一： 正常收敛到最小值
        if np.max(np.abs(extremum_update)) < 0.5: break
        # 更新新的最小可能坐标
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))
        # 超出边界了，那就是不稳定的特征，舍弃
        if min(i, image_shape[0] - i) < image_border_width \
                or min(j, image_shape[1] - j) < image_border_width \
                or image_index < 1 or image_index > num_intervals:
            return None
        # 如果到达了迭代最大值，那么也停止搜索了
        if attempt_index >= max_iter - 1:
            return None
        # 更新情况
        pixel_cube = dog_layers[image_index - 1:image_index + 2, i - 1:i + 2, j - 1:j + 2]
        gradient = cal_Grad(pixel_cube)
        hessian = cal_Hessian(pixel_cube)
        extremum_update = - np.linalg.lstsq(hessian, gradient, rcond=None)[0]

    ValueExtremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
    # 如果满足一定条件下，就需要更新，认为是最小点。这边主要为了去除边界的影响
    # 首先，同样要满足这边的最大点，要满足比threshold大
    if abs(ValueExtremum) >= contrast_threshold / num_intervals:
        xy_hessian = hessian[:2, :2]
        # 这里去除边缘效应。
        if np.linalg.det(xy_hessian) > 0 and (np.trace(xy_hessian) ** 2) / np.linalg.det(xy_hessian) < \
                ((eigenvalue_ratio + 1) ** 2) / eigenvalue_ratio:
            # 保存Keypoints!
            keypoint = cv2.KeyPoint()
            keypoint.pt = (  # 这边需要还原到原来的图像里面的坐标
                (j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
            # 我也不知道这里在干嘛？？？直接看的opencv抄的。。。
            keypoint.octave = octave_index + image_index * (2 ** 8) + \
                              int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            # 这个是图像相对于原图的总的模糊系数（scale）,用于后续确定度量gaussian 金字塔中的选取梯度范围的问题
            keypoint.size = keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(num_intervals))) \
                                            * (2 ** (octave_index + 1))
            # octave_index + 1 because the input image was doubled
            keypoint.response = abs(ValueExtremum)
            return keypoint, image_index
    return None


def com_Orientations(keypoint, octave_index, gaussian_image, num_bins=36):
    """计算主方向

    输入：
        :param keypoint: 计算得到的关键点，需要计算主方向的特征点
        :param octave_index: 所在的金字塔层数
        :param gaussian_image: 高斯金字塔
        :param num_bins: 梯度直方图的间隔。一般取 36 或 10
    输出：
        :return: 带有主方向的keypoints!

    """
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    # 确定边界大小。需要保证$3\sigma$区域内部可以被计算到
    scale = 1.5 * keypoint.size / np.float32(2 ** (octave_index + 1))
    radius = int(round(3 * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = np.zeros(num_bins)

    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i
        if 0 < region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j
                if 0 < region_x < image_shape[1] - 1:
                    # 如此，这里的区域一定是在图片里面，才能进行统计
                    # 计算梯度
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                    magnitude = np.sqrt(dx * dx + dy * dy)  # 计算梯度的模长，去除光照效应
                    orientation = np.rad2deg(np.arctan2(dy, dx))  # 计算梯度方向。
                    histogram_index = int(round(orientation * num_bins / 360.))
                    # 计算梯度，以及进行方向的计算。经过高斯加权后放入
                    raw_histogram[histogram_index % num_bins] += \
                        np.exp(weight_factor * (i ** 2 + j ** 2)) * magnitude

    smooth_histogram = np.zeros(num_bins)
    # 循环插值技术，光滑直方图
    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] +  # 最中间占6倍数
                               4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) +  # %用于循环插值。两边为4
                               raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.  # %用于循环插值。最外层为1

    orientation_max = max(smooth_histogram)
    orientation_peaks = np.where(np.logical_and(  # 找出直方图内部的局部极值点
        smooth_histogram > np.roll(smooth_histogram, 1),
        smooth_histogram > np.roll(smooth_histogram, -1)))[0]

    # 确定主方向流程
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= 0.8 * orientation_max:
            # 通过插值技术，把所在的离散方向还原出来。（注意这边直方图实际上是一个区间）
            # 这里使用了二次插值技术。三个点可以确定一个二次函数，然后进行
            # 最大化运算，找到最大值所在位置（很简单的数学推导，这里直接给出）
            left_value = smooth_histogram[(peak_index - 1) % num_bins]  # 注意边界情况需要用到取模运算
            right_value = smooth_histogram[(peak_index + 1) % num_bins]  # 注意边界情况需要用到取模运算
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (
                    left_value - 2 * peak_value + right_value)) % num_bins
            # 把算出来的主方向放入keypoint中（还有辅助方向）
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)
    return keypoints_with_orientations


def descriptors(keypoints, gaussian_images, window_width=4, num_bins=8,
                descriptor_max_value=0.2):
    """生成特征向量
    输入:
        :param keypoints: 特征点。
        :param gaussian_images: 高斯金字塔，用于梯度计算
        :param window_width: $4\times 4$的窗口，来自原作者推荐
        :param num_bins: 八个方向，用于计算每一个小矩阵上面的直方图情况
        :param descriptor_max_value: 超参数，用于过滤没用的特征

    输出：
        :return: 特征向量
    """
    descriptors = []

    for keypoint in keypoints:
        # 开始提取每一个特征. 这代码是人写的？？？？？？？？，好吧，opencv...
        octave = keypoint.octave & 255
        layer = (keypoint.octave >> 8) & 255
        if octave >= 128:
            octave = octave | -128
        scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
        # 回到那个图片的位置
        gaussian_image = gaussian_images[octave + 1][layer, :, :]
        num_rows, num_cols = gaussian_image.shape
        point = np.round(scale * np.array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint.angle
        # 提取旋转角度
        cos_angle = np.cos(np.deg2rad(angle))
        sin_angle = np.sin(np.deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)

        # 构建直方图代码
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        # first two dimensions are increased by 2 to account for border effects
        # 这边用来统计每一块地方的方向直方图，保障旋转不变性
        histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))

        # 同样的，用这一个$3\sigma$窗口进行度量统计
        hist_width = 3 * 0.5 * scale * keypoint.size
        half_width = int(round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))
        # sqrt(2) ，注意这边实际上使用的是圆形，那么边长自然是对角线的长度
        half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))

        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                # x, y 旋转后的坐标
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                # 同样也是旋转后的坐标，只不过把东西放到了对应的直方图区域内.
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                if -1 < row_bin < window_width and -1 < col_bin < window_width:
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))
                    if 0 < window_row < num_rows - 1 and 0 < window_col < num_cols - 1:
                        dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        # 计算梯度与方向，作为特征放入直方图内部
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                        weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        # 这边是旋转后的坐标，这边其实也说明了，我们插值的地方实际上是边界点。我们后续需要进行内插值，把对象还原出来
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        for i in range(row_bin_list.__len__()):
            # 使用逆变换方法对于插值结果进行一个平滑操作。
            row_bin = row_bin_list[i]
            col_bin = col_bin_list[i]
            magnitude = magnitude_list[i]
            orientation_bin = orientation_bin_list[i]

            # 还原到作保点，取floor就是为了对应到方格上的坐标点。
            row_bin_floor, col_bin_floor, orientation_bin_floor = \
                np.floor([row_bin, col_bin, orientation_bin]).astype(int)
            row_fraction, col_fraction, orientation_fraction = \
                row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            # 这里用到了三维插值的技术，具体可以见报告的示意图。
            # 其实也简单，这边magnitude是梯度模长，代表了权重
            # (1 - row_fraction)代表了从该点到前边row的边长(1 - col_fraction)则是列，(1 - orientation_fraction)这是高
            # 相当于把中心点的值，按照其比例权重，返回到各个边上，按照距离远近进行中心插值
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += \
                magnitude * (1 - row_fraction) * (1 - col_fraction) * (1 - orientation_fraction)
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += \
                magnitude * (1 - row_fraction) * (1 - col_fraction) * orientation_fraction
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += \
                magnitude * (1 - row_fraction) * col_fraction * (1 - orientation_fraction)
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += \
                magnitude * (1 - row_fraction) * col_fraction * orientation_fraction
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += \
                magnitude * row_fraction * (1 - col_fraction) * (1 - orientation_fraction)
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += \
                magnitude * row_fraction * (1 - col_fraction) * orientation_fraction
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += \
                magnitude * row_fraction * col_fraction * (1 - orientation_fraction)
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += \
                magnitude * row_fraction * col_fraction * orientation_fraction

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()
        # 最后进行归一化操作，减少光照对于特征的影响。
        threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(np.linalg.norm(descriptor_vector), float_tolerance)
        descriptor_vector = np.round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return np.array(descriptors, dtype='float32')
