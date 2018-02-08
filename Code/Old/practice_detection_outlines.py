import cv2
import numpy as np

cam = cv2.VideoCapture('../res/KishRace6BoatCloseShort.mp4')
count = 0
while count < 100:
    #img = cv2.imread('res/shapes.png', cv2.CV_8UC1)
    ret, img = cam.read()
    count+=1
cv2.imwrite('detection.png', img)
img = cv2.imread('detection.png', cv2.CV_8UC1)
#img = 255-img
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
adaptive_gray = cv2.medianBlur(img,5)
#blurred = cv2.GaussianBlur(gray, (11,11), 0)thresh = cv2.threshold(gray, 194, 15, cv2.THRESH_BINARY_INV)[1]
#
thresh = cv2.adaptiveThreshold(adaptive_gray,15, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 1)

thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)

cv2.imshow('thresh', thresh)
img2, cnts, h = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
    if cv2.contourArea(c) > 600:
        continue
    cv2.drawContours(img, c, -1, (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(2000)


