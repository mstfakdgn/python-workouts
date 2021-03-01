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

#convert np array to float

gray_flath_chess = np.float32(gray_flath_chess)

dst = cv2.cornerHarris(src=gray_flath_chess, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)
#to whow corners on real image
flath_chess[dst >0.01*dst.max()] = [255,0,0] #RGB
plt.imshow(flath_chess)
plt.show()

gray_real_chess = np.float32(gray_real_chess)

dst2 = cv2.cornerHarris(gray_real_chess, blockSize=2, ksize=3, k=0.04)
dst2 = cv2.dilate(dst2, None)
#to whow corners on real image
real_chess[dst2 >0.01*dst2.max()] = [255,0,0] #RGB
plt.imshow(real_chess)
plt.show()