import cv2

import numpy as np
import matplotlib.pyplot as plt


def display_image(img):
    fig = plt.figure(figsize=[12,10])
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap="gray")
    plt.show()

img = cv2.imread('../DATA/sudoku.jpg', 0)
display_image(img)

#x gradient
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
display_image(sobelx)

#y gradient
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
display_image(sobely)

#blend x and y
blended = cv2.addWeighted(src1=sobelx, alpha=0.5, src2=sobely, beta=0.5, gamma = 0)
display_image(blended)

#treshold
ret,th1 = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
display_image(th1)

#laplacian gradient
laplacian = cv2.Laplacian(img, cv2.CV_64F)
display_image(laplacian)

# morphological gradient
kernel = np.ones((5,5), np.uint8)
gradient = cv2.morphologyEx(blended, cv2.MORPH_GRADIENT, kernel)
display_image(gradient)

