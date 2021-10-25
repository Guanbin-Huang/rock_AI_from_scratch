import cv2

# link: https://docs.opencv.org/3.4.15/index.html


"""
1.Load an image
"""

img = cv2.imread("lena.jpg", 0)

print(img)

cv2.imshow("image", img)
k = cv2.waitKey(0) & 0xFF # ref: https://blog.csdn.net/hao5119266/article/details/104173400

if k == 27:
    cv2.destroyAllWindows()
    print("The esc has been pressed.")
# cv2.destroyWindow()
elif k == ord("s"):
    cv2.imwrite("lena_copy.png", img)
    cv2.destroyAllWindows()
    print("the lena image has been saved.")

else:
    print("other keys are pressed.")



