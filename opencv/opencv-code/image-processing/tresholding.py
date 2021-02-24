import cv2

import numpy as np
import matplotlib.pyplot as plt

def show_pic(img):
    fig = plt.figure(figsize = [15,15])
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()

# img = cv2.imread('../DATA/rainbow.jpg', 0)

# plt.imshow(img, cmap='gray')
# plt.show()
# ret1, thresh1 = cv2.threshold(img,127, 255, cv2.THRESH_BINARY)
# ret2, thresh2 = cv2.threshold(img,127, 255, cv2.THRESH_BINARY_INV)
# ret3, thresh3 = cv2.threshold(img,127, 255, cv2.THRESH_TRUNC)
# ret4, thresh4 = cv2.threshold(img,127, 255, cv2.THRESH_TOZERO)
# ret5, thresh5 = cv2.threshold(img,127, 255, cv2.THRESH_TOZERO_INV)
# plt.imshow(thresh1, cmap='gray')
# plt.show()
# plt.imshow(thresh2, cmap='gray')
# plt.show()
# plt.imshow(thresh3, cmap='gray')
# plt.show()
# plt.imshow(thresh4, cmap='gray')
# plt.show()
# plt.imshow(thresh5, cmap='gray')
# plt.show()

img = cv2.imread('../DATA/crossword.jpg', 0)
show_pic(img)
ret, thresh = cv2.threshold(img,180, 255, cv2.THRESH_BINARY)
show_pic(thresh)


# adaptive tresholding
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)
show_pic(th2)

th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8)
show_pic(th3)

blended = cv2.addWeighted(src1=thresh, alpha=0.6, src2=th2, beta=0.4, gamma=0)
show_pic(blended)

