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

blured_pennies = cv2.medianBlur(sep_coins, 35)
display(blured_pennies)

gray_blured_pennies = cv2.cvtColor(blured_pennies, cv2.COLOR_BGR2GRAY)
display(gray_blured_pennies)


#otsu thresholding
ret1, sep_thresh = cv2.threshold(gray_blured_pennies,0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
display(sep_thresh)

#NOISE REMOVAL
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(sep_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
display(opening)

sure_bg = cv2.dilate(opening,kernel,iterations=3)
display(sure_bg)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
display(dist_transform)


ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
display(sure_fg)


sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg, sure_fg)
display(unknown)


ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
display(markers)


markers = cv2.watershed(sep_coins, markers)
display(markers)

image, contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    
    # last column in the array is -1 if an external contour (no contours inside of it)
    if hierarchy[0][i][3] == -1:
        
        # We can now draw the external contours from the list of contours
        cv2.drawContours(sep_coins, contours, i, (255,0,0), 10)

display(sep_coins)