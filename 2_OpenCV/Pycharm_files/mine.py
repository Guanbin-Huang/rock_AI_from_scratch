import cv2
import matplotlib.pyplot as plt
'''
The following is the self-made resize method.
'''

import numpy as np

image = cv2.imread("hxw_250x176.jpg")

scale_x, scale_y = 1.5, 1.5
resized = cv2.resize(image, (0, 0), fx=scale_x, fy=scale_y)
cv2.imwrite("official_resize.jpg", resized)

src_h, src_w = image.shape[:2]
dst_h, dst_w = resized.shape[:2]

dst = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)

M = np.array([
    [scale_x, 0, 0],
    [0, scale_y, 0]
])

invert_M = cv2.invertAffineTransform(M)
const_value = 114

for dy in range(dst_h):
    for dx in range(dst_w):
        ix, iy = invert_M @ np.array([dx, dy, 1]) + np.array([0, 0]).T

        # 开始双线性插值，获取ix, iy最近的4个像素并插值

        ix_low = np.floor(ix)
        ix_high = ix_low + 1

        iy_low = np.floor(iy)
        iy_high = iy_low + 1

        # p0    p1
        #
        # p2    p3  可能出现 所要插值的点恰好压线

        if (ix == ix_low or ix == ix_high) and (iy != iy_low or iy != iy_high):  # ix happens to be int
            # only care about the top value and bottom value. ix = ix_low = ix_high
            weights = [
                (iy_high - iy),  # top weight
                (iy - iy_low)  # bottom weight
            ]

            position = [
                [ix_low, iy_low],  # top
                [ix_low, iy_high]  # bottom
            ]

            # print(f"w:{weights} \n pos: {position}")

        elif (ix != ix_low or ix != ix_high) and (iy == iy_low or iy == iy_high):  # iy happens to be int
            # only care about the left value and right value. iy = iy_low = iy_high
            weights = [
                (ix_high - ix),  # left weight
                (ix - ix_low)  # right weight
            ]

            position = [
                [ix_low, iy_low],  # left
                [ix_high, iy_low]  # right
            ]
            # print(f"w:{weights} \n pos: {position}")

        elif (ix == ix_low or ix == ix_high) and (iy == iy_low or iy == iy_high):  # both ix and iy happen to be int
            weights = [np.array([1.0])]

            position = [
                [ix_low, iy_low]
            ]
            # print(f"w:{weights} \n pos: {position}")

        else:
            weights = [
                (ix_high - ix) * (iy_high - iy),  # p0_weight
                (ix - ix_low) * (iy_high - iy),  # p1_weight
                (ix_high - ix) * (iy - iy_low),  # p2_weight
                (ix - ix_low) * (iy - iy_low)  # p3_weight
            ]

            position = [
                [ix_low, iy_low],   # p0
                [ix_high, iy_low],  # p1
                [ix_low, iy_high],  # p2
                [ix_high, iy_high]  # p3
            ]

        value = np.array([0, 0, 0], dtype=np.float32)

        # 筛选出未越界的点（越界了的点将该点的weight 平均分配给未越界的点）以确保weight 之和为 1
        valid_points_with_weight = [[(x, y), weight] for (x, y), weight in zip(position, weights)
                                    if x >= 0 and x < src_w and y >= 0 and y < src_h]

        if len(valid_points_with_weight) != 0:
            valid_points, weight = zip(*valid_points_with_weight)
            # print(weight)
            smooth_weight = 1 - sum(weight)
            weight = weight + smooth_weight / len(valid_points)
            # print(f"----\n {weight}")

            for (x, y), weight in zip(valid_points, weight):
                x,y = int(x), int(y)
                value += image[y, x] * weight  # 像素值 * 权重

            dst[dy, dx] = value.clip(0, 255)

        else:  # 如果所有点都越界了，那直接将该点的值置为0
            value = 0
            dst[dy, dx] = 0

cv2.imwrite("image1.jpg", dst)
print(dst.shape, dst.dtype)
resized = cv2.imread("./official_resize.jpg")
print(np.max(resized - dst))
# plt.imshow(dst[..., ::-1])
# plt.show()
