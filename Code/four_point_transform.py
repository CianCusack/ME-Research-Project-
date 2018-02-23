import cv2
import numpy as np
import matplotlib.pyplot as plt


def four_point_transform(imgs, mode):
    images = []
    for count, img in enumerate(imgs):
        h,w = img.shape[:2]
        img = cv2.resize(img, (400, 400))
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 140, 255, mode)[1]

        _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,  cv2.THRESH_BINARY_INV)
        if len(cnts) == 0:
            return None
        temp_x = []
        temp_y = []
        for c in cnts:
            #cv2.drawContours(img, c, -1, (0,255,0), 2)

            for c1 in c:
                temp_x.append(c1[0][0])
                temp_y.append(c1[0][1])

        #find min x and min y
        min_y = 400
        min_x = min(temp_x)
        list_x = list(temp_x)
        for index, p in enumerate(list_x):
            if p == min_x:
                if min_y > temp_y[index]:
                    min_y = temp_y[index]
        top_left = [min_x, min_y]

        # find  max y and min x
        max_y = max(temp_y)
        min_x = 400
        list_y = list(temp_y)
        for index, p in enumerate(list_y):
            if p == max_y:
                if min_x > temp_x[index]:
                    min_x = temp_x[index]
        bottom_left = [min_x, max_y]

        #find min y and max x
        min_y = min(temp_y)
        max_x = 0
        for index, p in enumerate(list_y):
            if p == min_y:
                if max_x < temp_x[index]:
                    max_x = temp_x[index]

        top_right = [max_x, min_y]

        #find max x and max y
        max_y = 0
        max_x = max(temp_x)
        for index, p in enumerate(list_x):
            if p == max_x:
                if max_y < temp_y[index]:
                    max_y = temp_y[index]

        bottom_right = [max_x, max_y]

        #cv2.rectangle(img, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]), (255,0,0), 2)
        # cv2.circle(img, (top_left[0], top_left[1]), 2, (255,0,0), 2)
        # cv2.circle(img, (top_right[0], top_right[1]), 2, (255, 0, 0), 2)
        # cv2.circle(img, (bottom_right[0], bottom_right[1]), 2, (255, 0, 0), 2)
        # cv2.circle(img, (bottom_left[0], bottom_left[1]), 2, (255,0,0), 2)

        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        transform_points = np.float32([top_left, bottom_left, top_right, bottom_right])
        transform_to_points = np.float32([[0, 0], [0,400], [400, 0], [400,400]])

        M = cv2.getPerspectiveTransform(transform_points,transform_to_points)
        dst = cv2.warpPerspective(img,M,(400,400))

        row, col= dst.shape[:2]
        bottom= dst[row-2:row, 0:col]

        # bordersize=15
        # dst=cv2.copyMakeBorder(dst, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[220,220,220] )

        # plt.subplot(121),plt.imshow(img),plt.title('Input')
        # plt.subplot(122),plt.imshow(dst),plt.title('Output')
        # plt.show()

        dst = cv2.resize(dst, (w,h))
        cv2.imwrite('../res/Sail Numbers/individual/rotated/{}.png'.format(count), dst)
        images.append(dst)
    return images
