import cv2
import matplotlib.pyplot as plt
import numpy as np

def display(img, cmap='gray'):
    fig = plt.figure(figsize=[12,10])
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()

sep_coins = cv2.imread('../DATA/pennies.jpg')
display(sep_coins)

#1->Median Blur
#2->Grayscale
#3->Binary Threshold


#1
blured = cv2.medianBlur(sep_coins, 25)
display(blured)

#2
gray = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY)
display(gray)

#3
ret1, sep_thresh = cv2.threshold(gray,160, 255, cv2.THRESH_BINARY_INV)
display(sep_thresh)


#contours
image,contours,hierarchy = cv2.findContours(sep_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(sep_coins, contours, i, (255,0,0), 10)

display(sep_coins)

## All not enought to seperate coins 

