import cv2
import numpy as np

# mouse callback function
def buoy_points(event,x,y,flags,param):
    global xCoord, yCoord
    if event == cv2.EVENT_LBUTTONDOWN:
        print x,y
        xCoord= x
        yCoord =  y


yCoord = 100
xCoord = 100

def record():
    cam = cv2.VideoCapture('res/KishRaceCloseBoat.mp4')
    display = cv2.namedWindow('image')
    cv2.setMouseCallback('image', buoy_points)
    while True:
        try:
            (ver, frame) = cam.read()

            cv2.rectangle(frame, (xCoord-26, yCoord-26), ((xCoord + 26), (yCoord + 26)), (255, 0, 0), 1)
            cv2.line(frame, (1280 / 2, 720), (xCoord, yCoord+26), (0,0,255), 1)
            roi = frame[yCoord - 25: yCoord + 25, xCoord - 25: xCoord + 25]

            #Edge detection
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.Canny(roi, 100,200)
            #Contours
            ret, thresh = cv2.threshold(roi, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, 1, 2)

            for c in contours:
                if cv2.contourArea(c) > 5000:
                    cv2.drawContours(roi, c, -1, (0,255,0), 3)

            cv2.imshow('ROI', roi)
            cv2.imshow('image', frame)

            cv2.waitKey(1)
        except:
            print "Reached end of file"
            cv2.destroyAllWindows()
            exit(-1)

