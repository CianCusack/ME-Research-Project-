import cv2

img = cv2.imread('res/High-Res Boats At A Distance/IMG_0214.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.Canny(img, 100, 500)




cv2.imshow('img', img)
cv2.waitKey(1000)