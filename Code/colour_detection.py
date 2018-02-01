import numpy as np
import cv2


def track_buoy_by_colour(frame):
    lower_bound = np.array([0, 0, 50])
    upper_bound = np.array([50, 50, 255])
    x,y,w,h, = 0,0,0,0
    # create a mask based on lower and upper boundaries
    mask = cv2.inRange(frame, lower_bound, upper_bound)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # get bounding rectangle of max contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y+h), (0,225,255), thickness=2)

    #z = frame[y - h:y + h, x - w:x + w].copy()
    return x, y, x+w, y+h
