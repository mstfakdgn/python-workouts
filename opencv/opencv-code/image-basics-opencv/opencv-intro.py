import numpy as np
import matplotlib.pyplot as plt

import cv2

img = cv2.imread('../DATA/00-puppy.jpg')
print(type(img), img, img.shape)
plt.imshow(img)
plt.show()

## MATPLOTLIB -> RGB RED GREEN BLUE
## OPENCV -> BLUE GREEN RED

fixed_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(fixed_image)
plt.show()

img_grey = cv2.imread('../DATA/00-puppy.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(img_grey, cmap="gray")
plt.show()

img_resized = cv2.resize(fixed_image, (1000,400))
plt.imshow(img_resized)
plt.show()

w_ratio = 0.5
h_ratio = 0.5

img_resized_ratio = cv2.resize(fixed_image, (0,0), fixed_image,w_ratio, h_ratio)
plt.imshow(img_resized_ratio)
plt.show()

flipped_img = cv2.flip(fixed_image,0)
plt.imshow(flipped_img)
plt.show()

cv2.imwrite('totally_new.jpg', cv2.cvtColor(fixed_image, cv2.COLOR_RGB2BGR))

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.imshow(fixed_image)
plt.show()
