import cv2

import numpy as np
import matplotlib.pyplot as plt

def display_image(img, cmap=None):
    fig = plt.figure(figsize=[10,8])
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap=cmap)
    plt.show()

rainbow = cv2.imread('../DATA/rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)

img = rainbow

# grabing x and x coordinates of image
mask = np.zeros(img.shape[:2], np.uint8)
mask[300:400,100:400] = 255
plt.imshow(mask, cmap='gray')
plt.show()

masked_img = cv2.bitwise_and(img,img, mask=mask)
show_masked_img = cv2.bitwise_and(show_rainbow, show_rainbow, mask=mask)

plt.imshow(show_masked_img)
plt.show()

hist_mask_values_red = cv2.calcHist([rainbow], channels=[2], mask=mask, histSize=[256], ranges=[0,256])
plt.plot(hist_mask_values_red)
plt.title("RED HISTOGRAM FOR MASKED RAINBOW")
plt.show()
hist_values_red = cv2.calcHist([rainbow], channels=[2], mask=None, histSize=[256], ranges=[0,256])
plt.plot(hist_values_red)
plt.title("RED HISTOGRAM FOR ORIGINAL RAINBOW")
plt.show()


gorilla = cv2.imread('../DATA/gorilla.jpg', 0)
display_image(gorilla, cmap='gray')

hist_gorilla = cv2.calcHist([gorilla], channels=[0], mask=None, histSize=[256], ranges=[0,256])
plt.plot(hist_gorilla)
plt.title('Gorilla Histogram')
plt.show()

eq_gorilla = cv2.equalizeHist(gorilla)
display_image(eq_gorilla, cmap='gray')

hist_equalized_gorilla = cv2.calcHist([eq_gorilla], channels=[0], mask=None, histSize=[256], ranges=[0,256])
plt.plot(hist_equalized_gorilla)
plt.title('Equalized Gorilla Histogram')
plt.show()


colored_gorilla = cv2.imread('../DATA/gorilla.jpg')
show_colored_gorilla = cv2.cvtColor(colored_gorilla, cv2.COLOR_BGR2RGB)
display_image(show_colored_gorilla)

#to equalize this colored picture we will use hsv
hsv_gorilla = cv2.cvtColor(colored_gorilla, cv2.COLOR_BGR2HSV)

#hsv[:,:,0] -> hue, hsv[:,:,1] -> saturation, hsv[:,:,2] -> value  
# for histogram equalization we have to equalize value channel
hsv_gorilla[:,:,2] = cv2.equalizeHist(hsv_gorilla[:,:,2])
rgb_gorilla = cv2.cvtColor(hsv_gorilla, cv2.COLOR_HSV2RGB)
display_image(rgb_gorilla)


