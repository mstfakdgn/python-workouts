import cv2

import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('../DATA/dog_backpack.png')
img1_fized = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('../DATA/watermark_no_copy.png')
img2_fized = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

plt.imshow(img1_fized)
plt.show()
plt.imshow(img2_fized)
plt.show()
print(img1_fized.shape, img2_fized.shape)

img1_resized = cv2.resize(img1_fized, (1200,1200))
img2_resized = cv2.resize(img2_fized, (1200,1200))
plt.imshow(img1_resized)
plt.show()
plt.imshow(img2_resized)
plt.show()


blended = cv2.addWeighted(src1=img1_resized, alpha=0.5, src2=img2_resized, beta=0.5, gamma = 0)
plt.imshow(blended)
plt.show()


# small image on top of larger image (no blending)
small_image = cv2.resize(img2_fized, (600,600))
large_image = img1_fized

x_offset = 250
y_offset = 600

x_end = x_offset + small_image.shape[1]
y_end = y_offset + small_image.shape[0]

large_image[y_offset:y_end, x_offset:x_end] = small_image
plt.imshow(large_image)
plt.show()


# blend together with different sizes
small_blend_image = cv2.resize(img2_fized, (600,600))

x_offset = 934 - 600
y_offset = 1401 - 600

rows,columns,channels = small_blend_image.shape

img1 = cv2.imread('../DATA/dog_backpack.png')
img1_fized = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

roi = img1_fized[y_offset:1401, x_offset:934]
plt.imshow(roi)
plt.show()

#create mask
small_blend_image_gray = cv2.cvtColor(small_blend_image,cv2.COLOR_RGB2GRAY)
plt.imshow(small_blend_image_gray, cmap="gray")
plt.show()

mask_inv = cv2.bitwise_not(small_blend_image_gray)
plt.imshow(mask_inv, cmap="gray")
plt.show()

white_background = np.full(small_blend_image.shape, 255, dtype=np.uint8)

bk = cv2.bitwise_or(white_background, white_background, mask=mask_inv)
plt.imshow(bk)
plt.show()

fg = cv2.bitwise_or(small_blend_image,small_blend_image,mask=mask_inv)
plt.imshow(fg)
plt.show()

final_roi = cv2.bitwise_or(roi,fg)
plt.imshow(final_roi)
plt.show()

large_img = img1_fized
small_img = final_roi

large_img[y_offset:y_offset+small_img.shape[0], x_offset:x_offset+small_img.shape[1]] = small_img

plt.imshow(large_img)
plt.show()
