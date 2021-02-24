import cv2

import numpy as np
import matplotlib.pyplot as plt

def load_img():
    img = np.zeros((600,600))
    return img

def display_image(img):
    fig = plt.figure(figsize=[12,10])
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap="gray")
    plt.show()

def write_picture(img,writing):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text=writing, org=(50, 300), fontFace=font,
            fontScale=5, color=(255, 255, 255), thickness=25, lineType=cv2.LINE_AA)
    return img

img = load_img()
img = write_picture(img, "ABCDE")
display_image(img)


#Erosion
kernel = np.ones((5,5), dtype=np.uint8)
result = cv2.erode(img,kernel, iterations=1)
result2 = cv2.erode(img,kernel, iterations=4)
display_image(result)
display_image(result2)



#add noise
img = load_img()
img = write_picture(img, "ABCDE")
white_noise = np.random.randint(low=0, high=2, size=(600,600))
white_noise = white_noise * 255
display_image(white_noise)

noised_img = img + white_noise
display_image(noised_img)


##Removing noise
opening = cv2.morphologyEx(noised_img, cv2.MORPH_OPEN, kernel)
display_image(opening)


#forgrand noise addding and cleaning

kernel = np.ones((5,5), dtype=np.uint8)

img = load_img()
img = write_picture(img, "ABCDE")
black_noise = np.random.randint(low=0, high=2, size=(600,600))
black_noise = black_noise * -255

black_noised_img = black_noise + img

black_noised_img[black_noised_img == -255] = 0

display_image(black_noised_img)

closing = cv2.morphologyEx(black_noised_img, cv2.MORPH_CLOSE, kernel)
display_image(closing)


gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT, kernel)
display_image(gradient)