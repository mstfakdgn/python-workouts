import cv2
import matplotlib.pyplot as plt
import numpy as np

def display_image(img):
    fig = plt.figure(figsize=[12,10])
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()

car = cv2.imread('../DATA/car_plate.jpg',0)

plate_cascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_licence_plate_rus_16stages.xml')

def detect_plate(img):
    car_img = car.copy()

    car_rects = plate_cascade.detectMultiScale(car_img)
    
    for (x,y,w,h) in car_rects:
        ROI = car_img[y:y+h, x:x+w]
        blured = cv2.medianBlur(ROI, 9)
        car_img[y:y+h, x:x+w] = blured
    
    return car_img

result = detect_plate(car)
display_image(result)