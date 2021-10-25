import cv2
import numpy as np
# events = [i for i in dir(cv2) if "EVENT" in i]
# print(events)

def click_event(event, x, y, flags, param): # event, x and y come from outside. img varaible can be captured from outside
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ",", y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x) + ', ' + str(y)
        cv2.putText(img, strXY, (x, y), font, 1, (255, 0, 0), 2)
        cv2.imshow("image", img)


    if event == cv2.EVENT_RBUTTONDOWN:
        blue = img[y, x, 0] # bgr
        green = img[y, x, 1]
        red = img[y, x, 2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        strBGR = str(blue) + ", " + str(green) + ", " + str(red)
        cv2.putText(img, strBGR, (x, y), font, .5, (255, 255, 0), 2)
        cv2.imshow("image", img) # 2



# img = np.zeros((512, 512, 3), dtype = np.uint8)
img = cv2.imread("lena.jpg")
cv2.imshow("image", img) # 1

cv2.setMouseCallback("image", click_event) # 3 a callback function is listening for the event of the mouse.
# the 3 "imshow" should be the same.

cv2.waitKey(0)
cv2.destroyAllWindows()
