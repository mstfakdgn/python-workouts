import cv2
import matplotlib.pyplot as plt
import numpy as np

def display_img(img,cmap=None):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)
    plt.show()

# img = cv2.imread('../DATA/giraffes.jpg')
# fixed_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# display_img(fixed_image)

# img = cv2.imread('../DATA/giraffes.jpg',0)
# ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# display_img(thresh1,cmap='gray')

# img = cv2.imread('../DATA/giraffes.jpg')
# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# display_img(img)

# kernel = np.ones(shape=(5,5),dtype=np.float32)/10
# img = cv2.imread('../DATA/giraffes.jpg')
# fixed_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# blured = cv2.filter2D(fixed_image, -1, kernel)
# display_img(blured)

# img = cv2.imread('../DATA/giraffes.jpg',0)
# sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
# display_img(sobelx,cmap='gray')


img = cv2.imread('../DATA/giraffes.jpg')

# three color histograms
color = ('b', 'g', 'r')

for i,col in enumerate(color):
    histr = cv2.calcHist([img], [i], mask=None, histSize=[256], ranges=[0,256])
    plt.plot('x', col,histr, color=col)
    plt.xlim([0,256])

plt.title('HISTOGRAM FOR GİRAFFES')
plt.legend()
plt.show()