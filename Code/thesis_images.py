import cv2


im = cv2.imread('../../Thesis/images/university.jpg')
h,w = im.shape[:2]
grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(grey, 127, 255, 0)
_, contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for c in contours:
    if cv2.contourArea(c) > (0.9*h)*(0.9*w):
        continue
    cv2.drawContours(im, c, -1,(0,0,255), 2)
cv2.imwrite('../../Thesis/images/ucd_contours.png', im)
cv2.imshow('im', im)
cv2.waitKey(0)