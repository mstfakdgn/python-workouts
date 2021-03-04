import numpy as np
import cv2
import time

corner_track_params = dict(maxCorners=10, qualityLevel=0.3,minDistance=7,blockSize=7)
lk_params = dict(winSize=(200,200), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))

cap = cv2.VideoCapture('../DATA/finger_move.mp4')

ret, prev_frame = cap.read()

prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# POINTS TO TRACK
prevPts = cv2.goodFeaturesToTrack(prev_frame_gray,mask=None, **corner_track_params)

mask = np.zeros_like(prev_frame)

while True:
    
    # Grab current frame
    ret,frame = cap.read()
    
    # Grab gray scale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate the Optical Flow on the Gray Scale Frame
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, frame_gray, prevPts, None, **lk_params)

    if nextPts is None:
        prevPts = cv2.goodFeaturesToTrack(prev_frame_gray, mask = None, **corner_track_params)
        nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, frame_gray, prevPts, None, **lk_params)
    
    # Using the returned status array (the status output)
    # status output status vector (of unsigned chars); each element of the vector is set to 1 if
    # the flow for the corresponding features has been found, otherwise, it is set to 0.
    good_new = nextPts[status==1]
    good_prev = prevPts[status==1]
    
    # Use ravel to get points to draw lines and circles
    for i,(new,prev) in enumerate(zip(good_new,good_prev)):
        
        x_new,y_new = new.ravel()
        x_prev,y_prev = prev.ravel()
        
        # Lines will be drawn using the mask created from the first frame
        mask = cv2.line(mask, (x_new,y_new),(x_prev,y_prev), (0,255,0), 3)
        
        # Draw red circles at corner points
        frame = cv2.circle(frame,(x_new,y_new),8,(0,0,255),-1)
    
    # Display the image along with the mask we drew the line on.
    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
   
    # Now update the previous frame and previous points
    prev_frame_gray = frame_gray.copy()
    prevPts = good_new.reshape(-1,1,2)
    
    
cv2.destroyAllWindows()
cap.release()