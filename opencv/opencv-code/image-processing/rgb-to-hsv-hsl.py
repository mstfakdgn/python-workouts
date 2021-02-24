import numpy as np
import matplotlib.pyplot as plt

import cv2

img = cv2.imread('../DATA/00-puppy.jpg')
plt.imshow(img)
plt.show()

fixed_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(fixed_image)
plt.show()

fixed_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
plt.imshow(fixed_image)
plt.show()

fixed_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
plt.imshow(fixed_image)
plt.show()