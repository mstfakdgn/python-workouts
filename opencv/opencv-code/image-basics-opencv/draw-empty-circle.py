import numpy as np
import matplotlib.pyplot as plt

import cv2

drawing = False
ix, iy = -1,-1

def draw_circle(event,x,y,flags,params):
    global ix,iy,drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event ==  cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(blank_img, center=(ix,iy), radius = x-ix, color=(0,255,0) ,thickness=-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(blank_img, center=(ix,iy), radius = x-ix, color=(0,255,0) ,thickness=-1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(blank_img, center=(x,y), radius= 15, color=(0,255,255), thickness=2)


blank_img = np.zeros(shape=(512,512,3))


cv2.namedWindow(winname='my_drawing')
cv2.setMouseCallback('my_drawing', draw_circle)


while True:
    cv2.imshow('my_drawing', blank_img)
    if cv2.waitKey(20) & 0xFF ==27:
        break

cv2.destroyAllWindows()