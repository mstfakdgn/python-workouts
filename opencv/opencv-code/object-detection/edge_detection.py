import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('../DATA/sammy_face.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

edges = cv2.Canny(image=img, threshold1=127, threshold2=127)
plt.imshow(edges)
plt.show()

#to find the optimum tresholds
median_value =np.median(img)

#lower treshold to either 0 or 70% of the median value whichever is greater
lower_treshold = int(max(0, 0.7*median_value))
# upper threshold to either 130% of the median or the max 255, whichever is smaller
upper_treshold = int(min(255, 1.3*median_value))

edges = cv2.Canny(image=img, threshold1=lower_treshold, threshold2=upper_treshold+200)
plt.imshow(edges)
plt.show()


# bluring for removing noise
blurred_img = cv2.blur(img, ksize=(7,7))
edges = cv2.Canny(image=blurred_img, threshold1=lower_treshold, threshold2=upper_treshold+50)
plt.imshow(edges)
plt.show()