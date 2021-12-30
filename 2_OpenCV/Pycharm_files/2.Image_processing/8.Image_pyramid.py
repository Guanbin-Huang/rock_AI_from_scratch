import cv2
img = cv2.imread("lena.jpg")
img_layer = img.copy()
g_pyramid = [img_layer]

# ref: https://paperswithcode.com/method/laplacian-pyramid

def make_shape_even(img):
    h, w = img.shape[:2]
    if h % 2 != 0:
        img = cv2.copyMakeBorder(img, 1,0,0,0, cv2.BORDER_REFLECT) # top, bottom, left, right

    if w % 2 != 0:
        img = cv2.copyMakeBorder(img, 0,0,1,0, cv2.BORDER_REFLECT)

    return img


# construct gaussian pyramid
for i in range(3):

    img_layer = cv2.pyrDown(img_layer)
    print(img_layer.shape)

    g_pyramid.append(img_layer) # the img_layer appended has to be even.
    cv2.imshow(str(i), img_layer)

img_layer = g_pyramid[-1]
l_pyramids = [img_layer]

# construct laplacian pyramid
for i in range(3, 0, -1):
    gaussian_expanded = cv2.resize(cv2.pyrUp(g_pyramid[i]), dsize = g_pyramid[i - 1].transpose(1,0,2).shape[:2]) # a trick
    laplacian = cv2.subtract(g_pyramid[i - 1], gaussian_expanded)
    cv2.imshow(str(i), laplacian)

cv2.imshow("Original image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
