import numpy as np
import cv2


def track_buoy_by_colour(frame, lower_bound, upper_bound):
    x,y,w,h, = 0,0,0,0
    # create a mask based on lower and upper boundaries
    mask = cv2.inRange(frame, lower_bound, upper_bound)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cv2.imshow('mask', mask)

    _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # get bounding rectangle of max contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y+h), (0,225,255), thickness=2)

    #z = frame[y - h:y + h, x - w:x + w].copy()
    return x, y, x+w, y+h


def track_objects_by_colour(frame, lower_bound, upper_bound):
        x, y, w, h, = 0, 0, 0, 0
        # create a mask based on lower and upper boundaries
        mask = cv2.inRange(frame.copy(), lower_bound, upper_bound)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cv2.imshow('mask', mask)

        _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                          cv2.THRESH_BINARY_INV)


        for c in contours:
            cv2.drawContours(frame, c, -1, (255,255,15), thickness=2)


def track_objects_by_colour_experimental(frame, lower_bound, upper_bound):
    x, y, w, h, = 0, 0, 0, 0

    h1,w1 = frame.shape[:2]
    frame1 = frame#[(3*h1)/4:h1, 0:w1]
    #frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    # create a mask based on lower and upper boundaries
    mask = cv2.inRange(frame1.copy(), lower_bound, upper_bound)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.resize(mask, (400, 400))
    cv2.imshow('mask', mask)

    _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                      cv2.THRESH_BINARY_INV)


cam = cv2.VideoCapture(0)
#cam = cv2.VideoCapture('../res/new_race.mov')
#cam = cv2.VideoCapture('../res/sailing.mov')
#cam = cv2.VideoCapture('../res/KishRace1.mp4')
cam = cv2.VideoCapture('../res/new_race_1.mov')
red = np.uint8([[[0,0,20]]])
light_red = np.uint8([[[150,150,255]]])
lower_bound = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
upper_bound = cv2.cvtColor(light_red, cv2.COLOR_BGR2HSV)
lower_bound = np.array([0, 0, 20])
upper_bound = np.array([100, 100, 255])
while True:

    ret, frame = cam.read()
    if not ret:
        break
    frame = cv2.imread('../res/High-Res Boats At A Distance/boats.jpg')
    frame = cv2.imread('../res//sail_numbers.jpg')
    #frame = cv2.resize(frame, (400,400))
    w, h = frame.shape[:2]
    #frame = frame[int((h*3)/4):h, 0:w]
    track_buoy_by_colour(frame, lower_bound, upper_bound)
    frame = cv2.resize(frame, (400,400))
    cv2.imshow('frame', frame)
    cv2.waitKey(50)