import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("thing.jpg", 0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kernel = np.ones((10,10), np.float32) / 100 # Low pass filter  and  high pass filter
dst = cv2.filter2D(img, -1, kernel) # -1 meaning:  ref: https://theailearner.com/tag/cv2-filter2d/#:~:text=If%20it%20is%20negative%2C%20it%20will%20be%20the%20same%20as%20that%20of%20the%20input%20image
blur = cv2.blur(img, (10, 10)) # mean_kernel.jpg
gblur = cv2.GaussianBlur(img, ksize = (11, 11), sigmaX=5, sigmaY=5) # gaussian_kernel.jpg
median = cv2.medianBlur(img, 5) # solver for salt and pepper noise
# homework： bilateralFilter = cv2.bilateralFilter(img, 50, 100, 100) # https://docs.opencv.org/4.5.3/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
# ref: https://www.youtube.com/watch?v=SWpnJh3RVUY

titles = ["image", "2D convolution", "blur", "Gaussian Blur","median","bilateralFilter"]
images = [img, dst, blur, gblur, median]

# plt.figure(figsize=(20,20))
for i in range(5):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()