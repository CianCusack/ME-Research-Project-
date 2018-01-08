import cv2
from train_haar_cascaade import *

cam = cv2.VideoCapture('../res/KishRace6BoatCloseShort.mp4')

while(1):
    #ret, img = cam.read()
    frame = cv2.imread('../../Presentation/Pictures/sailing-start-line.jpg')

    boats = use_cascade(frame.copy())
    for (x, y, w, h) in boats:
        cv2.circle(frame, (x, y + h), 2, (255, 0, 0), 2)
        cv2.line(frame, (x, y + h), (x + (w / 2), y), (0, 0, 255))
        cv2.circle(frame, (x + (w / 2), y), 2, (255, 0, 0), 2)
        cv2.line(frame, (x + (w / 2), y), (x + w, y + h), (0, 0, 255))
        cv2.circle(frame, (x + w, y + h), 2, (255, 0, 0), 2)
        cv2.line(frame, (x, y + h), (x + w, y + h), (0, 0, 255))
    frame = cv2.resize(frame, (1280,720))
    cv2.imshow('image', frame)
    #cv2.imwrite('../res/contour_example.png', img)
    cv2.waitKey(1000/23)
cam.release()