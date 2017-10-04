import cv2
import numpy as np
from SimpleCV import *

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global xCoord, yCoord
    if event == cv2.EVENT_LBUTTONDOWN:
        print x,y
        xCoord= x
        yCoord =  y


yCoord = 100
xCoord = 100

def record():
    e1 = cv2.getTickCount()
    cam = VirtualCamera('res/KishRace6.mp4', 'video')
    display = cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)
    counter = 1
    while True:
        try:
            #Convert simplecv capture to opencv output <3
            img = cam.getImage()
            bitmap = img.getBitmap()
            nump = np.asarray(bitmap[:,:])

            cv2.rectangle(nump, (xCoord-25, yCoord-25), ((xCoord + 25), (yCoord + 25)), (255, 0, 0), 3)
            cv2.line(nump, (img.width / 2, img.height), (xCoord, yCoord), (0,0,255), 1)
            roi = nump[yCoord - 25: yCoord + 25, xCoord - 25: xCoord + 25]
            #Edge detection
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            #roi = cv2.GaussianBlur(roi, (21, 21), 0)
            roi = cv2.Canny(roi, 100,200)

            #Contours
            ret, thresh = cv2.threshold(roi, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, 1, 2)
            for c in contours:
                if cv2.contourArea(c) > 50000:
                    cv2.drawContours(roi, c, -1, (0,255,0), 3)

            if counter%2==0:
                cv2.imshow('ROI', roi)
                cv2.imshow('image', nump)

            cv2.waitKey(1)
            counter = counter + 1

            e2 = cv2.getTickCount()
            time = (e2 - e1) / cv2.getTickFrequency()
            #print time
        except:
            print "Reached end of file"
            cv2.destroyAllWindows()
            exit(-1)

