import cv2
import matplotlib.pyplot as plt
import numpy as np

full = cv2.imread('../DATA/sammy.jpg')
full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)

plt.imshow(full)
plt.show()

sammy_face = cv2.imread('../DATA/sammy_face.jpg')
sammy_face = cv2.cvtColor(sammy_face, cv2.COLOR_BGR2RGB)

plt.imshow(sammy_face)
plt.show()

# All the 6 methods for comparison in a list
# Note how we are using strings, later on we'll use the eval() function to convert to function
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for m in methods:

    #create a copy of image
    full_copy = full.copy()

    method =eval(m)

    #template matching
    res = cv2.matchTemplate(full_copy, sammy_face, method)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else: 
        top_left = max_loc

    height, width, channels = sammy_face.shape

    bottom_right = (top_left[0] + width, top_left[1] + height)

    #draw on image and display
    cv2.rectangle(full_copy, top_left, bottom_right, (0,0,255), 10)

    #plot and show
    plt.subplot(121)
    plt.imshow(res)
    title = 'HEATMAP OF TEMPLATE MATCHING method =>' + m
    plt.title(title)
    plt.show()

    plt.subplot(122)
    plt.imshow(full_copy)
    title = 'DETECTION method =>'+m
    plt.title(title)
    plt.show()
