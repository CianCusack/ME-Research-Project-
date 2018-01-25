import cv2
from Buoy_Detection import *
from train_haar_cascaade import *
from motion_detection import *
from boat_detector import *
from imutils import contours
from line_crossing import *
import time

def read_video():
    #cam = cv2.VideoCapture(0)
    #cam = cv2.VideoCapture('../res/sailing.mov')
    #cam = cv2.VideoCapture('../res/olympic_sailing_short.mp4')
    #cam = cv2.VideoCapture('../res/new_race.mov')
    cam = cv2.VideoCapture('../res/KishRace6BoatCloseShort.mp4')
    display = cv2.namedWindow('image')
    cv2.setMouseCallback('image', buoy_points)
    ver, first = cam.read()
    #first = imutils.rotate(first, 90)
    cv2.imshow('image', first)
    cv2.waitKey(3000)
    cv2.destroyWindow('image')
    buoy = []
    boats = []
    counter = 0
    frame_counter = 1
    boat_count_spacer = 0
    last_x1, last_y1, last_x2, last_y2 = 0.00, 0.00, 0.00, 0.00
    t0 = time.time()
    filename = '../res/finishes.txt'
    file = open(filename, "w")
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        #Buoy
        #Only do operations every nth frame
        if frame_counter % 25 != 0:
            w,h = frame.shape[:2]
            buoy_x1, buoy_y1, buoy_x2, buoy_y2, buoy = track_buoy(frame.copy(), buoy)
            m = slope((w/2, h), (buoy_x1, buoy_y2))
            max_buoy_width = max_buoy_height = 60
            if buoy_x1 != 0 and buoy_y1 != 0:
                if(abs(buoy_x1 - last_x1) <= max_buoy_width and abs(buoy_y1 - last_y1) <= max_buoy_height) or last_y1 == 0:
                    w,h = frame.shape[:2]
                    cv2.rectangle(frame, (int(buoy_x1), int(buoy_y1)), (int(buoy_x2), int(buoy_y2)), (0,255,0), 1)
                    cv2.line(frame, (w / 2, h), (int(buoy_x1), int(buoy_y2)), (0, 0, 255), 1)
                    last_x1, last_y1, last_x2, last_y2 = buoy_x1, buoy_y1, buoy_x2, buoy_y2
            else:
                cv2.putText(frame, "No buoy", (200, 300), cv2.FONT_HERSHEY_COMPLEX_SMALL, 10, (0, 0, 0))
                counter += 1
                if counter > 10:
                    last_y1 = 0

            #Boats
            # boats = use_cascade(frame.copy())

            boats, coords = detect_boats(frame)


            if boats != []:
                for i, c in enumerate(coords):
                    # bottom_left = (c[0], c[3])
                    # bottom_right = (c[2], c[3])
                    # top_middle = (c[0]+(c[2] - c[0])/2, c[1])
                    # cv2.circle(frame, (bottom_left), 2, (255, 0, 0), 2)
                    # cv2.circle(frame, (top_middle), 2, (255, 0, 0), 2)
                    # cv2.circle(frame, (bottom_right), 2, (255, 0, 0), 2)
                    # cv2.line(frame, bottom_left, top_middle, (0, 0, 255))
                    # cv2.line(frame, top_middle, bottom_right, (0, 0, 255))
                    # cv2.line(frame, bottom_left, bottom_right, (0, 0, 255))
                    y_points = np.arange(c[1], c[3], 0.01)
                    for p in y_points:
                        if p > buoy_y1:
                            m1 = slope((c[2], p), (buoy_x1, buoy_y2))
                            #print 'line slope: {}, boat {} slope: {} for ({},{})'.format(m, i+1, m1, c[2], p)
                            if m1 == m:
                                cv2.putText(frame, "Intersection", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,100), 2)
                                cv2.imwrite('../res/Screen-Shots/line_crossing.png', frame)
                                file.write('Boat {} finished at{} \n'.format(i, time.time() - t0))
                                #exit(1)
            #         boat_copy = locate_numbers(boat)
                #cv2.imshow("boat", boat)
        cv2.imshow('image', frame)
        cv2.waitKey(1)
        frame_counter += 1
    cam.release()
    file.close()

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