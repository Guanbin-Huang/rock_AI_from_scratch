import cv2
import matplotlib.pyplot as plt

img = cv2.imread("lena.jpg")

print(type(img))

cv2.imshow("image", img)
k = cv2.waitKey(0) & 0xFF

if k == 27:
    cv2.destroyAllWindows()
    print("the esc has been pressed")

elif k == ord("s"):
    cv2.imwrite("lena_copy.png", img)
    cv2.destroyAllWindows()
    print("the image has been saved")

else:
    print("other keys are pressed")
