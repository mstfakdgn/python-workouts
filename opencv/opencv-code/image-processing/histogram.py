import cv2

import numpy as np
import matplotlib.pyplot as plt

dark_horse = cv2.imread('../DATA/horse.jpg')
show_horse = cv2.cvtColor(dark_horse, cv2.COLOR_BGR2RGB)

rainbow = cv2.imread('../DATA/rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)

blue_bricks = cv2.imread('../DATA/bricks.jpg')
show_bricks = cv2.cvtColor(blue_bricks, cv2.COLOR_BGR2RGB)

# plt.imshow(show_horse)
# plt.show()
# plt.imshow(show_rainbow)
# plt.show()
# plt.imshow(show_bricks)
# plt.show()

#OPENCV BGR
 # forblue
hist_value_brick = cv2.calcHist([blue_bricks], channels=[0], mask=None, histSize=[256], ranges=[0,256])
plt.plot(hist_value_brick)
plt.title('brick')
plt.show()

hist_value_horse = cv2.calcHist([dark_horse], channels=[0], mask=None, histSize=[256], ranges=[0,256])
plt.plot(hist_value_horse)
plt.title('horse')
plt.show()



# three color histograms
color = ('b', 'g', 'r')

for i,col in enumerate(color):
    histr = cv2.calcHist([blue_bricks], [i], mask=None, histSize=[256], ranges=[0,256])
    plt.plot('x', col,histr, color=col)
    plt.xlim([0,256])

plt.title('HISTOGRAM FOR BLUE BRICKS')
plt.legend()
plt.show()


color = ('b', 'g', 'r')

for i,col in enumerate(color):
    histr = cv2.calcHist([dark_horse], [i], mask=None, histSize=[256], ranges=[0,256])
    plt.plot('x', col,histr, color=col)
    plt.xlim([0,50])
    plt.ylim([0,500000])

plt.title('HISTOGRAM FOR DARK HORSE')
plt.legend()
plt.show()