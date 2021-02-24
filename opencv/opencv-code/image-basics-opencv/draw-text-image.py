import numpy as np
import matplotlib.pyplot as plt

import cv2

blank_img = np.zeros(shape=(512, 512, 3), dtype=np.int16)
print(blank_img.shape)
plt.imshow(blank_img)
plt.show()

font = cv2.FONT_HERSHEY_SIMPLEX
# org -> bottom left corner
cv2.putText(blank_img, text='Hello', org=(10, 500), fontFace=font,
            fontScale=4, color=(255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
plt.imshow(blank_img)
plt.show()

vertices = np.array( [[100, 300], [200, 200], [400, 300], [200, 400]], dtype=np.int32)

#add another dimension because of opencv
pts = vertices.reshape((-1,1,2))
cv2.polylines(blank_img, [pts], isClosed=True, color=(255,0,0), thickness=5)
plt.imshow(blank_img)
plt.show()