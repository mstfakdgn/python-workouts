import cv2

cap = cv2.VideoCapture('../DATA/finger_move.mp4')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HIGHT))

# Top left corner
x = width // 2 # allows integer
y = height // 2

# width and height of rectangle
w = width // 4
h = height // 4

# bottom right corner


while True:
    ret, frame = cap.read()
    cv2.rectangle(frame, (x,y), (x+w, y+h), color = (0,0,255), thickness=4)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()