import numpy as np
import struct
import random
import matplotlib.pyplot as plt
from numpy.core.arrayprint import _extendLine
import pandas as pd
import math
import cv2

def conv(image, kernel):
    # refer to ipad 2.Conv
    # 输入的信息
    in_h, in_w, in_channel = image.shape # info of the image
    k_size, _ = kernel.shape                # 最简易版本里，所有通道都用相同的卷积核
    st_offset = k_size // 2                
    end_offset = -(k_size // 2)
    padding = 0
    stride = 1

    # 输出的信息
    out_h = int((in_h + 2 * padding - k_size) / stride + 1) #todo 推导
    out_w = int(in_w + 2 * padding - k_size / stride + 1) # (inclusive) img的左上角坐标直接加上start_offset就可以得到kernel在图片上的起点
    output = np.zeros((out_h, out_w, in_channel))    # (exclusive) img的右下角直接加上这个end_offset就可以得到kernel在图片上的终点

    for cy in range(st_offset, in_h + end_offset):  
        for cx in range(st_offset, in_w + end_offset):
            for k_row in range(k_size):
                for k_col in range(k_size):
                    output[cy - st_offset, cx - st_offset] += \
                        kernel[k_row, k_col] * image[cy - st_offset + k_row, cx - st_offset + k_col] # 从kernel左上角对应的地方image像素开始累加
    
    return output

def gaussian_kernel2d(size, sigma):
    # refer to gaussian.jpg
    s = 2 * sigma ** 2
    center = size // 2 # 3 // 2 = 1    4 // 2 = 2 (这里的2是idx  即 0 1 2 3)  5 // 2 = 2 (0 1 2 3 4)
    output = np.zeros((size, size))
    
    # 开始填入高斯产生的值
    for i in range(size):
        for j in range(size):
            y = i - center # i 偏离 center 的距离
            x = i - center 
            output[i,j] = np.exp(-( x**2 + y**2)) / s
    return output / np.sum(output)


image = cv2.imread("kitty.jpg")
image = cv2.resize(image, (100, 80))
lap_kernel = np.array([ # refer to 科普了解 https://blog.csdn.net/qq_38131594/article/details/80776367
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]
], dtype = np.float32) # astype(np.float32)

output = conv(image, lap_kernel)
output = (255*(output - np.min(output)) / (np.max(output) - np.min(output))).astype(np.uint8) # cv2.imwrite() 必须是 0-255 才能显示
# output = (255*(output - np.min(output)) / (np.max(output) - np.min(output))) # plt.imshow 接受的值域要么是0-1  要么是 0-255
cv2.imwrite("laplacian_conv_img.jpg", output)

gau_kernel = gaussian_kernel2d(7, 1)
output = conv(image, gau_kernel)
output = (255*(output - np.min(output)) / (np.max(output) - np.min(output))).astype(np.uint8) # cv2.imwrite() 必须是 0-255 才能显示
cv2.imwrite("gaussian_conv_img.jpg", output)





