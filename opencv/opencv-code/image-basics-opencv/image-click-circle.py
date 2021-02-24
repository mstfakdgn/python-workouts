import numpy as np
import matplotlib.pyplot as plt

import cv2

def draw_circle(event, x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(blank_img, (x,y), radius=50, color=(0,255,0), thickness=-1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(blank_img, (x,y), radius=50, color=(255,0,0), thickness=-1)

cv2.namedWindow(winname='my_drawing')
cv2.setMouseCallback('my_drawing', draw_circle)

blank_img = np.zeros(shape=(512,512,3), dtype=np.int8)

while True:
    cv2.imshow('my_drawing', blank_img)
    if cv2.waitKey(20) & 0xFF ==27:
        break

cv2.destroyAllWindows()