import cv2
import numpy as np

img = cv2.imread('res/Screen-Shots/Intersection1.png')
norm_img = cv2.normalize(img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
g_blur = cv2.GaussianBlur(gray, (21, 21), 0)

v = np.median(img)
#apply automatic Canny edge detection using the computed median
lower = int(max(0, (1.0 - 0.33) * v))
upper = int(min(255, (1.0 + 0.33) * v))
canny = cv2.Canny(gray, lower, upper)

kernel = np.ones((5,5), np.uint8)

img_dilation = cv2.dilate(canny, kernel, iterations=1)
#cv2.imshow('Dilation', img_dilation)
#cv2.waitKey(5000)

th, im_th = cv2.threshold(img_dilation, 220, 255, cv2.THRESH_BINARY_INV);
im_floodfill = im_th.copy()
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(im_floodfill, mask, (0,0), 255);
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
#cv2.imshow('flood fill', im_floodfill_inv)
#cv2.waitKey(5000)
im_out = img_dilation | im_floodfill_inv


img_erosion = cv2.erode(im_floodfill_inv, kernel, iterations=1)

(cnts, _) = cv2.findContours(img_erosion.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
c = max(cnts, key = cv2.contourArea)


cv2.drawContours(img_erosion, c, -1, 255, -1)

res = cv2.bitwise_and(img,img,mask = img_erosion)
res = img - res
res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
cv2.imwrite("res/Screen-Shots/Result.png", res)
#res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
(cnts, _) = cv2.findContours(res.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
    if cv2.contourArea(c) < 700:
        continue
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

(cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
for c in cnts:
    if cv2.contourArea(c) < 700:
        continue
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


cv2.imwrite("res/Screen-Shots/Contours.png", img)


#cv2.imshow('image', res)
#cv2.waitKey(5000)