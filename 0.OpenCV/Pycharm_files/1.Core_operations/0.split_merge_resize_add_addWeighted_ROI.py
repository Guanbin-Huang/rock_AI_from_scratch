import numpy as np
import cv2

img = cv2.imread("messi.jpg")
img2 = cv2.imread("lena.jpg")

print(img.shape)
print(img.size)
print(img.dtype)

b, g, r = cv2.split(img)
merged_img = cv2.merge((b,g,r))

ball = img[228:257, 252:302]
img[228:257, 100:150] = ball

img = cv2.resize(img, (512, 512))
img2 = cv2.resize(img2, (512, 512))

# dst = cv2.add(img, img2) # color is distorted
dst = cv2.addWeighted(img, 0.7, img2, 0.3, 0) # ref: https://docs.opencv.org/3.4.15/d2/de8/group__core__array.html#gafafb2513349db3bcff51f54ee5592a19

cv2.imshow("image", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
# xyrb 252 228 302 257
