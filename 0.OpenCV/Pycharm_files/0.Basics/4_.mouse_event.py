

import cv2

def click_event(event, x, y, flags, param):  # event, x, y come from outside.
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ",", y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x) + ',' + str(y)
        cv2.putText(img, strXY, (x, y), font, 1, (255, 0, 0), 2)
        cv2.imshow("image", img)

    elif event == cv2.EVENT_RBUTTONDOWN:
        blue  = img[y, x, 0] # bgr bgr bgr
        green = img[y, x, 1]
        red   = img[y, x, 2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        strBGR = str(blue) + ", " + str(green) + "," + str(red)
        cv2.putText(img, strBGR, (x, y), fontFace= 0, fontScale=.5, color = (255, 255, 0), thickness=2)
        cv2.imshow("image", img)

    pass


img = cv2.imread("lena.jpg") # read image
cv2.imshow("image", img)     # show image

cv2.setMouseCallback("image", click_event) # set a callback function. Pass a function into it.

cv2.waitKey(0)  # wait for a key
cv2.destroyAllWindows()
