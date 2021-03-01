import cv2
import matplotlib.pyplot as plt
import numpy as np

flath_chess = cv2.imread('../DATA/flat_chessboard.png')
flath_chess = cv2.cvtColor(flath_chess, cv2.COLOR_BGR2RGB)
plt.imshow(flath_chess)
plt.show()

gray_flath_chess = cv2.cvtColor(flath_chess, cv2.COLOR_RGB2GRAY)
plt.imshow(gray_flath_chess, cmap='gray')
plt.show()

real_chess = cv2.imread('../DATA/real_chessboard.jpg')
real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)
plt.imshow(real_chess)
plt.show()

gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_RGB2GRAY)
plt.imshow(gray_real_chess, cmap='gray')
plt.show()

#second parameter number of the corners we want to detect
corners = cv2.goodFeaturesToTrack(gray_flath_chess, 64, 0.01, 10)

corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(flath_chess, (x,y), 3, (255,0,0),-1)

plt.imshow(flath_chess)
plt.show()


corners = cv2.goodFeaturesToTrack(gray_real_chess, 100, 0.01, 10)

corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(real_chess, (x,y), 3, (255,0,0),-1)

plt.imshow(real_chess)
plt.show()