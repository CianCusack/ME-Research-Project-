import cv2
import numpy as np

w, h = 1920,1080
w1, h1 = 6*80,int(4.2*80)
blank_image = np.zeros((h,w,3), np.uint8)
blank_image[0:h1,0:w1] = (255,0,0)
img = blank_image #cv2.resize(blank_image, (1280, 720))
cv2.imshow('img', img)
cv2.waitKey(0)




