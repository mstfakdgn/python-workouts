import cv2
import matplotlib.pyplot as plt
import numpy as np

flath_chess = cv2.imread('../DATA/flat_chessboard.png')
plt.imshow(flath_chess)
plt.show()

found, corners = cv2.findChessboardCorners(flath_chess, (7,7))

#corners
cv2.drawChessboardCorners(flath_chess, (7,7), corners, found)
plt.imshow(flath_chess)
plt.show()


dots = cv2.imread('../DATA/dot_grid.png')
plt.imshow(dots)
plt.show()

found, corners = cv2.findCirclesGrid(dots, (10,10), cv2.CALIB_CB_SYMMETRIC_GRID)

#corners
cv2.drawChessboardCorners(dots, (10,10), corners, found)
plt.imshow(dots)
plt.show()