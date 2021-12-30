import cv2
import numpy as np

# def click_event(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
#         points.append((x, y))
#         if len(points) >= 2:
#             cv2.line(img, points[-1], points[-2], (255, 0, 0), 5)
#         cv2.imshow("image", img)
#
# img = cv2.imread("lena.jpg")
# cv2.imshow("image", img)
# points = []
# cv2.setMouseCallback("image", click_event)
#
# cv2.waitKey(0) # keep waiting until a key is pressed.
# cv2.destroyAllWindows()


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        blue  = img[x, y, 0]
        green = img[x, y, 1]
        red   = img[x, y, 2]
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        colorPlate = np.zeros((512, 512, 3), dtype = np.uint8) # dtype must be specified.

        colorPlate[:] = [blue, green, red]

        cv2.imshow("colorPicked", colorPlate)

img = cv2.imread("lena.jpg")
cv2.imshow("image", img)
cv2.setMouseCallback("image", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()


