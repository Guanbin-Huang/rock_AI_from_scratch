import cv2
import numpy as np

img = cv2.imread("sudoku.jpg", 0)
print(img.shape)
_, simple_thres = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
adaptive_thres = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2) # ref: https://docs.opencv.org/4.5.3/d7/d1b/group__imgproc__misc.html#:~:text=Enumerator-,ADAPTIVE_THRESH_MEAN_C%C2%A0,-Python%3A%20cv.ADAPTIVE_THRESH_MEAN_C

cv2.imshow("Image", img)
cv2.imshow("simple_thres", simple_thres)
cv2.imshow("adaptive_thres", adaptive_thres)



cv2.waitKey(0)
cv2.destroyAllWindows()