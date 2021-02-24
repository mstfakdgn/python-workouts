import cv2

import numpy as np
import matplotlib.pyplot as plt

def load_img():
    img = cv2.imread('../DATA/bricks.jpg').astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def display_image(img):
    fig = plt.figure(figsize=[12,10])
    ax = fig.add_subplot(111)
    ax.imshow(img)
    plt.show()

def write_picture(img,writing):
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(img, text=writing, org=(50, 600), fontFace=font,
            fontScale=10, color=(255, 0, 0), thickness=4, lineType=cv2.LINE_AA)
    return img


img = load_img()
display_image(img)


## Gamma
gamma = 1/4
gamma2 = 2

result = np.power(img, gamma)
result2 = np.power(img, gamma2)
display_image(result)
display_image(result2)

##Bluring
write_picture(img, 'brick')
display_image(img)

#bluring kernel how big blures more
kernel = np.ones(shape=(5,5),dtype=np.float32)/25
display_image(kernel)

blured = cv2.filter2D(img, -1, kernel)
display_image(blured)


#reset
img = load_img()
write_picture(img, 'default_kernel')
blured =  cv2.blur(img,ksize=(10,10))
display_image(blured)


# Gaussian
img = load_img()
write_picture(img, 'gaussian')
blured = cv2.GaussianBlur(img, (5,5), 10)
display_image(blured)


#Median uses for removing noises
img = load_img()
write_picture(img, 'median')
blured = cv2.medianBlur(img, 5)
display_image(blured)



img = cv2.imread('../DATA/sammy.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
display_image(img)


noise_img = cv2.imread('../DATA/sammy_noise.jpg')
noise_img = cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB)
display_image(noise_img)
blured = cv2.medianBlur(noise_img, 5)
display_image(blured)



#BilateralFiter
img = load_img()
write_picture(img, 'bilateral')
blured = cv2.bilateralFilter(img,9,75,75)
display_image(blured)