import cv2
import numpy as np
# apple = cv2.imread("apple.jpg")
# orange = cv2.imread("orange.jpg")
#
# apple_resized = cv2.resize(apple, (512, 512))
# orange_resized = cv2.resize(orange, (512, 512))
#
# cv2.imwrite("512x512_apple.jpg", apple_resized)
# cv2.imwrite("512x512_orange.jpg", orange_resized)

# the theory under the hood: ref: https://becominghuman.ai/image-blending-using-laplacian-pyramids-2f8e9982077f

apple = cv2.imread("512x512_apple.jpg")
orange = cv2.imread("512x512_orange.jpg")
apple_orange = np.hstack((apple[:, :256], orange[:, 256:]))

apple_copy = apple.copy()
gp_apple = [apple_copy]

# generate Gaussian pyramid for apple and orange respectively ----------------------------------------------------------
for i in range(6):
    apple_copy = cv2.pyrDown(apple_copy)
    gp_apple.append(apple_copy)

orange_copy = orange.copy()
gp_orange = [orange_copy]

for i in range(6):
    orange_copy = cv2.pyrDown(orange_copy)
    gp_orange.append(orange_copy)


# generate Laplacian pyramid for apple and orange respectively ---------------------------------------------------------
apple_copy = gp_apple[5]
lp_apple = [apple_copy]
for i in range(5, 0, -1):
    gaussian_expanded = cv2.pyrUp(gp_apple[i])
    laplacian = cv2.subtract(gp_apple[i - 1], gaussian_expanded)
    lp_apple.append(laplacian)

orange_copy = gp_orange[5]
lp_orange = [orange_copy]
for i in range(5, 0, -1):
    gaussian_expanded = cv2.pyrUp(gp_orange[i])
    laplacian = cv2.subtract(gp_orange[i - 1], gaussian_expanded)
    lp_orange.append(laplacian)

# Now add left and right halves of images in each level ----------------------------------------------------------------
apple_orange_pyramid = []
n = 0
for apple_lap, orange_lap in zip(lp_apple, lp_orange):
    n += 1
    cols, rows, ch = apple_lap.shape
    laplacian = np.hstack((apple_lap[:, 0:int(cols/2)], orange_lap[:, int(cols/2):]))
    apple_orange_pyramid.append(laplacian)
    # cv2.imshow(f"apple_orange_pyramid_{n}", laplacian)

# now reconstruct
apple_orange_reconstruct = apple_orange_pyramid[0]
for i in range(1, 6):
    apple_orange_reconstruct = cv2.pyrUp(apple_orange_reconstruct)
    apple_orange_reconstruct = cv2.add(apple_orange_pyramid[i], apple_orange_reconstruct)


cv2.imshow("blended", apple_orange_reconstruct)
cv2.waitKey(0)
cv2.destroyAllWindows()


