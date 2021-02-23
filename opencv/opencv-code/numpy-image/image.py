import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

pic = Image.open('../DATA/00-puppy.jpg')
pic_array = np.asarray(pic)
print(pic_array, pic_array.shape)
plt.imshow(pic_array)
plt.show()

pic_red = pic_array.copy()

plt.imshow(pic_array[:,:,0], cmap='gray')
plt.show()
plt.imshow(pic_array[:,:,1], cmap='gray')
plt.show()
plt.imshow(pic_array[:,:,2], cmap='gray')
plt.show()

#removed green
pic_red[:,:,1] = 0
plt.imshow(pic_red)
plt.show()

# remove also blue only red
pic_red[:,:,2] = 0
plt.imshow(pic_red)
plt.show()