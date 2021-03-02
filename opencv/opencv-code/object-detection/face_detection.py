import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

nadia = cv2.imread('../DATA/Nadia_Murad.jpg',0)
denis = cv2.imread('../DATA/Denis_Mukwege.jpg',0)
solvay= cv2.imread('../DATA/solvay_conference.jpg',0)

face_cascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_frontalface_default.xml')

def detect_face(img):
    face_img = img.copy()

    face_rects = face_cascade.detectMultiScale(face_img)

    for (x,y,w,h) in face_rects:
        ROI = image[y:y+h, x:x+w]
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10)
    
    return face_img

result_nadia = detect_face(nadia)
result_denis = detect_face(denis)
result_solvay = detect_face(solvay)

plt.imshow(result_nadia, cmap='gray')
plt.show()

plt.imshow(result_denis, cmap='gray')
plt.show()

plt.imshow(result_solvay, cmap='gray')
plt.show()


def adj_detect_face(img):
    face_img = img.copy()

    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)

    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10)
    
    return face_img

result_nadia = adj_detect_face(nadia)
result_denis = adj_detect_face(denis)
result_solvay = adj_detect_face(solvay)

plt.imshow(result_nadia, cmap='gray')
plt.show()

plt.imshow(result_denis, cmap='gray')
plt.show()

plt.imshow(result_solvay, cmap='gray')
plt.show()


eye_cascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_eye.xml')

def detect_eyes(img):
    face_img = img.copy()

    eye_rects = eye_cascade.detectMultiScale(face_img)

    for (x,y,w,h) in eye_rects:
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10)
    
    return face_img

result_nadia = detect_eyes(nadia)
result_denis = detect_eyes(denis)
result_solvay = detect_eyes(solvay)

plt.imshow(result_nadia, cmap='gray')
plt.show()

cap = cv2.VideoCapture('../DATA/video_capture.mp4')

if cap.isOpened() == False:
    print('FILE NOT FOUND')

while cap.isOpened():
    ret, frame = cap.read()

    if ret == True:

        # #writer 20 fps
        # time.sleep(1/20)

        frame = detect_face(frame)

        cv2.imshow('Video Fac Detect', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    else:
        break

cap.release()
cv2.destroyAllWindows()