import cv2
from Buoy_Detection import *
from train_haar_cascaade import *
from motion_detection import *

def read_video():
    #cam = cv2.VideoCapture('../res/horizontal_race.mov')
    #cam = cv2.VideoCapture('../res/new_race.mov')
    cam = cv2.VideoCapture('../res/KishRace6.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('demo.mp4', fourcc, 23.0, (1280, 720))
    display = cv2.namedWindow('image')
    cv2.setMouseCallback('image', buoy_points)
    ver, first = cam.read()
    #first = imutils.rotate(first, 90)
    cv2.imshow('image', first)
    cv2.waitKey(3000)
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
        buoy_x1, buoy_y1, buoy_x2, buoy_y2, buoy = track_buoy(frame.copy(), buoy)
        max_buoy_width = max_buoy_height = 60
        cv2.rectangle(frame, (buoy_x1, buoy_y1), (buoy_x2, buoy_y2), (0,255,0), 1)
        if (buoy_x1 != 1280 and buoy_y1 != 720 and (
                abs(buoy_x1 - last_x1) <= max_buoy_width and abs(buoy_y1 - last_y1) <= max_buoy_height)) or last_y1 == 0:
            cv2.rectangle(frame, (buoy_x1, buoy_y1), (buoy_x2, buoy_y2), (255, 0, 0), 1)
            cv2.line(frame, (1280 / 2, 720), (buoy_x1, buoy_y2), (0, 0, 255), 1)
            last_x1, last_y1, last_x2, last_y2 = buoy_x1, buoy_y1, buoy_x2, buoy_y2
        else:
            cv2.putText(frame, "No buoy", (200, 300), cv2.FONT_HERSHEY_COMPLEX_SMALL, 10, (0, 0, 0))
            counter += 1
            if counter > 10:
                last_y1 = 0

        #Boats
        boats = use_cascade(frame.copy())
        for (x, y, w, h) in boats:
            cv2.circle(frame, (x, y + h), 2, (255, 0, 0), 2)
            cv2.line(frame, (x, y + h), (x + (w / 2), y), (0, 0, 255))
            cv2.circle(frame, (x + (w / 2), y), 2, (255, 0, 0), 2)
            cv2.line(frame, (x + (w / 2), y), (x + w, y + h), (0, 0, 255))
            cv2.circle(frame, (x + w, y + h), 2, (255, 0, 0), 2)
            cv2.line(frame, (x, y + h), (x + w, y + h), (0, 0, 255))
        cv2.imshow('image', frame)
        out.write(frame)
        cv2.waitKey(1)
    cam.release()
    out.release()

