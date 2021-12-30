import cv2
import numpy as np

image = cv2.imread("hxw_250x176.jpg")
sx, sy = 1.5, 1.5
resized = cv2.resize(image, (0,0), fx = sx, fy = sy)
cv2.imwrite("official_resize.jpg", resized)
print(resized.shape[:2])

src_h, src_w = image.shape[:2]
dst_h, dst_w = resized.shape[:2]

dst = np.zeros((dst_h, dst_w, 3), dtype = np.uint8)

M = np.array([
    [sx, 0,   0],
    [0,  sy, 0],
])
M = cv2.getRotationMatrix2D((50, 50), 30, 1)

invert_M = cv2.invertAffineTransform(M)
const_value = 0

for dy in range(dst_h):
    for dx in range(dst_w):
        src_x, src_y = invert_M @ np.array([dx, dy, 1]) # + np.array([0.5, 0.5])

        if (src_x <= -1 or src_x >= src_w or src_y <= -1 or src_y >= src_h):
            dst[dy, dx, :] = const_value

        else:
            # 开始双线性插值 获取离ix, iy 最近的4个像素并插值
            y_low = int(np.floor(src_y))
            x_low = int(np.floor(src_x))
            y_high = int(y_low + 1)
            x_high = int(x_low + 1)

            ly = src_y - y_low
            lx = src_x - x_low
            hy = 1 - ly
            hx = 1 - lx


            w1, w2, w3, w4 = 0, 0, 0, 0
            v1, v2, v3, v4 = const_value, const_value, const_value, const_value
            if (y_low >= 0):
                if (x_low >= 0):
                    v1 = image[y_low, x_low, :]
                    w1 = hy * hx

                if (x_high < src_w):
                    v2 = image[y_low, x_high, :]
                    w2 = hy * lx

            if (y_high < src_h):
                if (x_low >= 0):
                    v3 = image[y_high, x_low, :]
                    w3 = ly * hx

                if (x_high < src_w):
                    v4 = image[y_high, x_high, :]
                    w4 = ly * lx

            wsum = w1 + w2 + w3 + w4 # w1    w2  w3  w4
            if wsum == 0:
                p = const_value
            else:
                scale = 1 / wsum
                w1 *= scale
                w2 *= scale
                w3 *= scale
                w4 *= scale
                p = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
            dst[dy, dx, :] = p


cv2.imwrite("mine.jpg", dst)




