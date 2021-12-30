import cv2

img = cv2.imread("opencv_logo.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


ret, thresh = cv2.threshold(img_gray, 127, 255, 0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print("Number of contours = " + str(len(contours)))

for c_idx in range(len(contours)):
    cv2.drawContours(img, contours, c_idx, (122, 0, 122), 3)

cv2.imshow("Image", img)
cv2.imshow("Image gray", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

