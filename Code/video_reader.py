import cv2
from Buoy_Detection import *
from train_haar_cascaade import *
from motion_detection import *
from boat_detector import *
from imutils import contours
from line_crossing import *

def read_video():
    #cam = cv2.VideoCapture(0)
    cam = cv2.VideoCapture('../res/sailing.mov')
    #cam = cv2.VideoCapture('../res/olympic_sailing_short.mp4')
    #cam = cv2.VideoCapture('../res/new_race.mov')
    #cam = cv2.VideoCapture('../res/KishRace6.mp4')
    display = cv2.namedWindow('image')
    cv2.setMouseCallback('image', buoy_points)
    ver, first = cam.read()
    #first = imutils.rotate(first, 90)
    cv2.imshow('image', first)
    #cv2.waitKey(3000)
    cv2.destroyWindow('image')
    buoy = []
    boats = []
    counter = 0
    boat_count_spacer = 0
    last_x1, last_y1, last_x2, last_y2 = 0, 0, 0, 0
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        #Buoy
        w,h = frame.shape[:2]
        m = slope(((w/2)+100, h), (1000, 400))
        cv2.line(frame, ((w / 2) +100, h), (1000, 400), (0, 0, 255), 1)
        # buoy_x1, buoy_y1, buoy_x2, buoy_y2, buoy = track_buoy(frame.copy(), buoy)
        # max_buoy_width = max_buoy_height = 60
        # cv2.rectangle(frame, (buoy_x1, buoy_y1), (buoy_x2, buoy_y2), (0,255,0), 1)
        # if buoy_x1 != 0 and buoy_y1 != 0:
        #     if(abs(buoy_x1 - last_x1) <= max_buoy_width and abs(buoy_y1 - last_y1) <= max_buoy_height) or last_y1 == 0:
        #         w,h = frame.shape[:2]
        #         cv2.rectangle(frame, (buoy_x1, buoy_y1), (buoy_x2, buoy_y2), (255, 0, 0), 1)
        #         cv2.line(frame, (w / 2, h), (buoy_x1, buoy_y2), (0, 0, 255), 1)
        #         last_x1, last_y1, last_x2, last_y2 = buoy_x1, buoy_y1, buoy_x2, buoy_y2
        # else:
        #     cv2.putText(frame, "No buoy", (200, 300), cv2.FONT_HERSHEY_COMPLEX_SMALL, 10, (0, 0, 0))
        #     counter += 1
        #     if counter > 10:
        #         last_y1 = 0

        #Boats
        # boats = use_cascade(frame.copy())
        # for (x, y, w, h) in boats:
        #     cv2.circle(frame, (x, y + h), 2, (255, 0, 0), 2)
        #     cv2.line(frame, (x, y + h), (x + (w / 2), y), (0, 0, 255))
        #     cv2.circle(frame, (x + (w / 2), y), 2, (255, 0, 0), 2)
        #     cv2.line(frame, (x + (w / 2), y), (x + w, y + h), (0, 0, 255))
        #     cv2.circle(frame, (x + w, y + h), 2, (255, 0, 0), 2)
        #     cv2.line(frame, (x, y + h), (x + w, y + h), (0, 0, 255))
        boats, coords = detect_boats(frame.copy())
        if boats != []:
            for c in coords:
                y_points = np.arange(c[1], c[3], 1)
                for p in y_points:
                    if slope((c[2], p), (1000, 400)) == m:
                        cv2.putText(frame, "Intersection", (100, 100),
			    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,100), 2)
        #         boat_copy = locate_numbers(boat)
            #cv2.imshow("boat", boat)
        cv2.imshow('image', frame)
        cv2.waitKey(1)
    cam.release()

def locate_numbers(boat):
    sensitivity = 100
    lower_white = np.array([0, 0, 255 - sensitivity])
    upper_white = np.array([255, sensitivity, 255])
    (h, w) = boat.shape[:2]
    boat_copy = boat.copy()
    boat = boat[h / 3:(2 * h) / 3, 0:w]
    hsv = cv2.cvtColor(boat, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(boat, boat, mask=mask)
    boat = cv2.cvtColor(boat, cv2.COLOR_BGR2GRAY)
    _, boat = cv2.threshold(boat, 75, 110, 0)
    _, cnts, _ = cv2.findContours(boat.copy(), cv2.RETR_LIST,
                                  cv2.CHAIN_APPROX_SIMPLE)

    cnts = contours.sort_contours(cnts, method="left-to-right")[0]
    rois = []
    for c in cnts:
        if cv2.contourArea(c) > 100 or cv2.contourArea(c) < 50:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # draw the book contour (in green)
        roi = boat[y:y + h, x:x + w].copy()
        roi = cv2.resize(roi, (200, 75))
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([0, 100, 100])
        mask = cv2.inRange(hsv, lower_black, upper_black)
        roi = cv2.bitwise_and(roi, roi, mask=mask)
        # roi = cv2.Canny(roi, 10, 30)
        rois.append(roi)
        cv2.rectangle(boat_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(boat_copy, c, -1, (0, 255, 0), 1)

    i = 0
    for roi in rois:
        cv2.imshow("roi {}".format(i), roi)
        i += 1
    cv2.waitKey(500)
    while i >= 0:
        cv2.destroyWindow("roi {}".format(i))
        i -= 1
    return boat_copy