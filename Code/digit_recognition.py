import cv2
import numpy as np

img = cv2.imread('../res/sail_numbers.jpg')

mser = cv2.MSER_create()

#Resize the image so that MSER can work better
img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
vis = img.copy()



regions = mser.detectRegions(gray)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
i = 0
for p in hulls:
    temp_x = []
    temp_y = []
    for p1 in p:
        temp_x.append(p1[0][0])
        temp_y.append(p1[0][1])
    if (max(temp_x) - (min(temp_x)) < 10 or
                (max(temp_y)) - min(temp_y)) < 35 or \
                            (max(temp_x) + max(temp_y)) - (min(temp_x) + min(temp_y)) > 300 or\
                        max(temp_x) - (min(temp_x)) > 100:
        continue

    cv2.rectangle(vis, (min(temp_x), min(temp_y)), (max(temp_x), max(temp_y)), (255, 12, 145), 4)
    #cv2.imwrite('../res/Sail Numbers/{}.png'.format(i), vis[min(temp_y):max(temp_y), min(temp_x):max(temp_x)])
    i += 1

cv2.namedWindow('img', 0)
cv2.imshow('img', vis)
cv2.imwrite('../res/sail_numbers_noise.png', vis)
cv2.destroyAllWindows()


