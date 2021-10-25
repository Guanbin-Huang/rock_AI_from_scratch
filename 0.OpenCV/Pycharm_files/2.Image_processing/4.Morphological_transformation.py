"""
- Simple operations on the image shape
- Normally performed on the binary image
- two things are required
    - original image
    - kernel( structuring element)
        - a kernel tells you how to change the value of any given pixel by
            combining it with different amounts of the neighboring pixels

ref: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html

"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("colorful_ball.jpg", 0)
_, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((3,3), np.uint8)

dilation = cv2.dilate(mask, kernel, iterations=1) # dilation-series.jpg   ref: https://www.youtube.com/watch?v=7-FZBgrW4RE&t=164s
erosion = cv2.erode(mask, kernel, iterations=2)
openning = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # erosion followed by dilation
closing  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # dilation followed by erosion
mg = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)

titles = ["image","mask", "dilation","erosion","openning","closing","mg"]
images = [img, mask, dilation, erosion, openning, closing, mg]

for i in range(7):
    plt.subplot(2, 4, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()


