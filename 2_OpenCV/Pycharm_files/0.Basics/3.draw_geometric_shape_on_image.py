import numpy as np
import cv2
img = cv2.imread("lena.jpg", 1)
print(img.shape)
img = cv2.line(img, (0,0), (100,100), (0, 255, 0), 2)
img = cv2.arrowedLine(img, (0, 50), (100, 50), (255, 0, 0), 2)
img = cv2.rectangle(img, (384, 0), (510, 128), (0,0,255), 10)
img = cv2.circle(img, (447, 63), 63, (0, 255, 0), -1) # -1: fill the circle    ref: https://www.geeksforgeeks.org/python-opencv-cv2-circle-method/

font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.putText(img, "OpenCV", (10, 500), font, 4, (0, 255, 0), 10, 16) # cv2.LINE_AA

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
