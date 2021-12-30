import numpy as np
import cv2

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cap.set(3, 300)
cap.set(4, 300)

cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)


while True:
    _, frame = cap.read()

    # frame = cv2.imread("colorful_ball.jpg")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # ref: https://dsp.stackexchange.com/questions/2687/why-do-we-use-the-hsv-colour-space-so-often-in-vision-and-image-processing#:~:text=67,HSV%20separates%20luma

    lh = cv2.getTrackbarPos("LH", "Tracking")
    ls = cv2.getTrackbarPos("LS", "Tracking")
    lv = cv2.getTrackbarPos("LV", "Tracking")

    uh = cv2.getTrackbarPos("UH", "Tracking")
    us = cv2.getTrackbarPos("US", "Tracking")
    uv = cv2.getTrackbarPos("UV", "Tracking")

    lower_bound = np.array([lh, ls, lv])
    upper_bound = np.array([uh, us, uv])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    res = cv2.bitwise_and(frame, frame, mask = mask)
    # mask = np.repeat(mask[...,None], 3, axis = 2)
    # res = mask & frame

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1) #!
    if key == 27 or key == ord("q"):
        break

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)


cap.release()
cv2.destroyAllWindows()