import cv2
import numpy as np
from feature_recognition_practice import *

# mouse callback function
def buoy_points(event,x,y,flags,param):
    global xCoord, yCoord
    if event == cv2.EVENT_LBUTTONDOWN:
        xCoord= x
        yCoord =  y


yCoord = 100
xCoord = 100


def record():
    cam = cv2.VideoCapture('../res/new_race_2.MOV')
    starting_frame = np.zeros((720,780,3), np.uint8)
    display = cv2.namedWindow('image')
    cv2.setMouseCallback('image', buoy_points)
    count = 0
    while True:
        try:
            (ver, frame) = cam.read()
            copy = frame.copy()

            #cv2.rectangle(frame, (xCoord-26, yCoord-26), ((xCoord + 26), (yCoord + 26)), (255, 0, 0), 1)
            cv2.line(frame, (1280 / 2, 720), (xCoord, yCoord), (0,0,255), 1)
            roi = copy[yCoord - 25: yCoord + 25, xCoord - 25: xCoord + 25]



            #Edge detection
            #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            #roi = cv2.Canny(roi, 20,50 ,  L2gradient=True)
            roi = cv2.resize(roi,(720,480))




            """ret, thresh = cv2.threshold(roi, 127, 255, 0)
            (_,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                # if the contour is too small, ignore it
                #if cv2.contourArea(c) < 500:
                 #   continue

                cv2.rectangle(roi, c, (c+5), (0, 255, 0), 2)"""

            if xCoord != 100 and yCoord != 100 and count ==0:
                starting_frame = roi.copy()
                count +=1
            if(xCoord!=100):
                #diff = cv2.absdiff(starting_frame, roi)
                cv2.imshow('Starting frame', starting_frame)
                cv2.imwrite('roi1.png', roi)
                cv2.imwrite('roi.png', starting_frame)

            cv2.imshow('ROI', roi)
            cv2.imshow('image', frame)

            cv2.waitKey(1)
        except:
            print "Reached end of file"
            cv2.destroyAllWindows()
            exit(-1)


def track_buoy():
    cam = cv2.VideoCapture('../res/new_race_4.mov')
    display = cv2.namedWindow('image')
    cv2.setMouseCallback('image', buoy_points)
    ver, first = cam.read()
    cv2.imshow('image', first)
    cv2.waitKey(3000)
    count = 0
    while True:
        if xCoord != 100 and yCoord != 100:

            (ver, frame) = cam.read()
            if not ver:
                break

            #Buoy
            if count < 1:
                buoy = frame[yCoord - 75:yCoord + 75, xCoord - 75:xCoord + 75].copy()
                hsv = cv2.cvtColor(buoy, cv2.COLOR_BGR2HSV)
                lower_red = np.array([150, 150, 100])
                upper_red = np.array([255, 255, 255])

                mask = cv2.inRange(hsv, lower_red, upper_red)
                res = cv2.bitwise_and(buoy, buoy, mask=mask)

                cv2.imshow('Detected buoy', buoy)
                cv2.imshow('masked buoy', res)
                count+=1
            #Frame
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_red, upper_red)
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

            #returns bounding points of buoy
            x1, y1, x2, y2 = match_features(res, masked_frame)
            #xCoord, yCoord = (x1+x2)/2, (y1+y2)/2
            #cv2.rectangle(frame, (xCoord-26, yCoord-26), ((xCoord + 26), (yCoord + 26)), (255, 0, 0), 1)
            if x1 != 1280 and y1 != 720:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.line(frame, (1280 / 2, 720), (x2, y2), (0, 0, 255), 1)
            else:
                cv2.putText(frame, "No buoy", (200, 300), cv2.FONT_HERSHEY_COMPLEX_SMALL, 10, (0,0,0))

            cv2.imshow('image', frame)
            cv2.waitKey(50)

    print "Reached end of file"
    cv2.destroyAllWindows()
    exit(-1)
