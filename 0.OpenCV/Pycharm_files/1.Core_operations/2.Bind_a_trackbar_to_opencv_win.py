import numpy as np
import cv2

def nothing(x):
    print(x)


# Create a black image, a win
img = np.zeros((300, 512, 3), dtype = np.uint8)
cv2.namedWindow("canvas_win")

# CREATE TRRACK BAR
cv2.createTrackbar("B", "canvas_win", 0, 255, nothing) # callback function that will be called.
cv2.createTrackbar("G", "canvas_win", 0, 255, nothing) # callback function that will be called.
cv2.createTrackbar("R", "canvas_win", 0, 255, nothing) # callback function that will be called.

switch = "0 : OFF\n 1 : ON"
cv2.createTrackbar(switch, "canvas_win", 0, 1, nothing)

while(True):
    cv2.imshow("canvas_win", img)
    key = cv2.waitKey(1) & 0xFF # https://stackoverflow.com/questions/51143458/difference-in-output-with-waitkey0-and-waitkey1/51143586#:~:text=From%20the%20doc,for%20image%20display).
    if key == 27:
        break

    b = cv2.getTrackbarPos("B", "canvas_win")
    g = cv2.getTrackbarPos("G", "canvas_win")
    r = cv2.getTrackbarPos("R", "canvas_win")
    s = cv2.getTrackbarPos(switch, "canvas_win")

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b, g, r]

cv2.destroyAllWindows()