import numpy as np
import cv2


def track_buoy_by_colour(frame, lower_bound, upper_bound):
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
    frame1 = frame[(3*h1)/4:h1, 0:w1]

    # create a mask based on lower and upper boundaries
    mask = cv2.inRange(frame1.copy(), lower_bound, upper_bound)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cv2.imshow('mask', mask)

    _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                      cv2.THRESH_BINARY_INV)
    # if len(contours) > 0:
    #     # get bounding rectangle of max contour
    #     c = max(contours, key=cv2.contourArea)
    #     x, y, w, h = cv2.boundingRect(c)
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 255), thickness=2)
    #     cv2.imshow('image', frame[y:y + h, x:x + w])

    temp = np.array([0, 0, 0])
    for c in contours:
        cv2.drawContours(frame1, c, -1, (255, 255, 15), thickness=2)
        temp += frame[c[0][0][0]][c[0][0][1]]
    print temp
    print len(contours)
    print ((temp / len(contours)) * 0.95)#, (temp / len(c) * 1.05)

cam = cv2.VideoCapture(0)
#cam = cv2.VideoCapture('../res/new_race.mov')
#cam = cv2.VideoCapture('../res/sailing.mov')
#cam = cv2.VideoCapture('../res/KishRace1.mp4')
cam = cv2.VideoCapture('../res/new_race_1.mov')
lower_bound = np.array([146, 131, 120])
upper_bound = np.array([186, 171, 160])
while True:

    ret, frame = cam.read()
    if not ret:
        break
    frame = cv2.imread('../res/test/test2.png')
    frame = cv2.resize(frame, (400,400))
    w, h = frame.shape[:2]
    #frame = frame[int((h*3)/4):h, 0:w]
    track_objects_by_colour_experimental(frame, lower_bound, upper_bound)
    cv2.imshow('frame', frame)
    cv2.waitKey(50)