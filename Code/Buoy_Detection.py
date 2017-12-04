import cv2
import numpy as np
import imutils
from feature_recognition_practice import *
from template_matching import *

yCoord = 100
xCoord = 100
count = 0
# mouse callback function
def buoy_points(event,x,y,flags,param):
    global xCoord, yCoord
    if event == cv2.EVENT_LBUTTONDOWN:
        xCoord= x
        yCoord =  y





def record():
    cam = cv2.VideoCapture('../res/horizontal_race.MOV')
    starting_frame = np.zeros((720,780,3), np.uint8)
    display = cv2.namedWindow('image')
    cv2.setMouseCallback('image', buoy_points)
    count = 0
    while True:
        try:
            (ver, frame) = cam.read()
            frame = imutils.rotate(frame, 90)
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



def track_buoy(frame, buoy = []):
    last_x1, last_y1, last_x2, last_y2 = 0,0,0,0
    buoy_points_array = []
    lower_red = np.array([150, 150, 100])
    upper_red = np.array([255, 255, 255])
    global count
    counter =0
    if xCoord != 100 and yCoord != 100:
        #Buoy
        if count < 1:
            distance = input('Approximately how far away is the buoy?')
            size = calc_range(distance)
            print size
            buoy = frame[yCoord - size:yCoord + size, xCoord - size:xCoord + size].copy()
            max_buoy_height, max_buoy_width = buoy.shape[:2]
            hsv = cv2.cvtColor(buoy, cv2.COLOR_BGR2HSV)


            mask = cv2.inRange(hsv, lower_red, upper_red)
            res = cv2.bitwise_and(buoy, buoy, mask=mask)
            cv2.imwrite('../res/buoy.png', buoy)
            cv2.imshow('Detected buoy', buoy)
            cv2.imshow('masked buoy', res)
            count+=1
        #Frame
        hsv = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_red, upper_red)
        masked_frame = cv2.bitwise_and(frame.copy(), frame.copy(), mask=mask)

        #returns bounding points of buoy
        #x1, y1, x2, y2 = match_features(res, masked_frame, last_x1, last_y1, last_x2, last_y2)
        x1, y1, x2, y2 =  match_template(frame, buoy)
        return x1,y1,x2,y2, buoy
        """buoy_points_array.append(x1)
        #buoy_points_array.append([(x1,y1), (x2,y2)])
        #xCoord, yCoord = (x1+x2)/2, (y1+y2)/2
        #cv2.rectangle(frame, (xCoord-26, yCoord-26), ((xCoord + 26), (yCoord + 26)), (255, 0, 0), 1)
        if (x1 != 1280 and y1 != 720 and (abs(x1-last_x1) <= max_buoy_width and abs(y1-last_y1)<=max_buoy_height)) or last_y1==0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.line(frame, (1280 / 2, 720), (x2, y2), (0, 0, 255), 1)
            last_x1, last_y1, last_x2, last_y2 = x1, y1, x2, y2
        else:
            cv2.putText(frame, "No buoy", (200, 300), cv2.FONT_HERSHEY_COMPLEX_SMALL, 10, (0,0,0))
            #cv2.rectangle(frame, (last_x1, last_y1), (last_x2, last_y2), (255, 0, 0), 1)
            #cv2.line(frame, (1280 / 2, 720), (last_x2, last_y2), (0, 0, 255), 1)
            counter+=1
            if counter > 10:
                last_y1 = 0"""
    else:
        return 0,0,0,0, buoy

def calc_range(distance):
    size = 20
    if distance > (2.5*size):
        if distance > (size * 5):
            return int(size - 1.5 * (distance / size))
        return int(size-4*(distance/size))

    else:
        return int(1.5*size)