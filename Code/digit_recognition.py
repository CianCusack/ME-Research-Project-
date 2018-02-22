# import cv2
# from train_digits import recognise_digits
# import numpy as np
#
# def find_digit_areas(img):
#     mser = cv2.MSER_create()
#     #Resize the image so that MSER can work better
#     img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     vis = img.copy()
#     regions = mser.detectRegions(gray)
#     hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
#     mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
#
#     for contour in hulls:
#         cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
#
#     # this is used to find only text regions, remaining are ignored
#     text_only = cv2.bitwise_and(img, img, mask=mask)
#
#     cv2.imshow("text only", text_only)
#
#     cv2.waitKey(0)
#     i = 0
#     imgs = []
#     for p in hulls:
#         temp_x = []
#         temp_y = []
#         for p1 in p:
#             temp_x.append(p1[0][0])
#             temp_y.append(p1[0][1])
#         temp_img = vis[min(temp_y):max(temp_y), min(temp_x):max(temp_x)]
#         reject = False
#         # for x in temp_img:
#         #     for t in x:
#         #         if t[0] == 255 and t[1] == 12 and t[2] == 145:
#         #             reject = True
#         #             break
#         #
#         # if reject == True:
#         #     continue
#
#
#         h,w = temp_img.shape[:2]
#         if w <  20 or h < 15:
#             continue
#
#         solid_black = np.zeros((h,w,3), np.uint8)
#         gray = cv2.cvtColor(temp_img.copy(), cv2.COLOR_BGR2GRAY)
#         _, thresh = cv2.threshold(gray.copy(), 75, 128, cv2.THRESH_BINARY_INV)
#         _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
#                                               cv2.THRESH_BINARY_INV)
#         for c in contours:
#             cv2.drawContours(solid_black, c, -1, (255, 0,0), thickness=2)
#         # cv2.rectangle(vis, (min(temp_x), min(temp_y)), (max(temp_x), max(temp_y)), (255, 12, 145), 1)
#         imgs.append(solid_black)
#         cv2.imwrite('../res/Sail Numbers/{}.png'.format(i), solid_black)#[min(temp_y):max(temp_y), min(temp_x):max(temp_x)])
#         i += 1
#         # cv2.imshow('p', solid_black)
#         # cv2.waitKey(0)
#         # cv2.destroyWindow('p')
#     return imgs
#
#
# img = cv2.imread('../res/sail_numbers.jpg')
# #w, h = img.shape[:2]
# #img = img[0:h, 0:2*w/3]
# # cv2.imshow('img', img)
# # cv2.waitKey(0)
#
# imgs = find_digit_areas(img)
#
# #imgs = [cv2.resize(im, (30,30)) for im in imgs]
# recognise_digits(imgs)

#!/usr/bin/python3
# 2017.10.05 10:52:58 CST
# 2017.10.05 13:27:18 CST
"""
Text detection with MSER, and fill with random colors for each detection.
"""

import numpy as np
import cv2
from four_point_transform import *
from recognise_numbers import recognise_digits
import difflib
import pandas as pd
import math

def detect_digits(img):
    h, w = img.shape[:2]
    img = img[0:3*h/5, 0:w]
    #img = cv2.resize(img, (w*2, h*2))
    #h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## Get mser, and set parameters
    mser = cv2.MSER_create()
    mser.setMinArea(60)
    mser.setMaxArea(200)

    ## Do mser detection, get the coodinates and bboxes
    coordinates, bboxes = mser.detectRegions(gray)
    ## Filter the coordinates
    coords = []
    boxes = []
    i = 0
    for coord in coordinates:
        bbox = cv2.boundingRect(coord)
        x,y,w1,h1 = bbox
        if w1< 8 or h1 < 15 or w1 > h1 or h1/w1 > 10:
            continue
        coords.append(coord)
        boxes.append(bbox)
        #cv2.rectangle(img, (int(x),int(y)), (int(x+w1), int(y+h1)), (100,100,100), 1)
        # rect = cv2.minAreaRect(coord)
        #
        # box = cv2.boxPoints(rect)
        #
        # box = np.int0(box)
        #
        # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    canvas3 = np.zeros([img.shape[0],img.shape[1],3],dtype=np.uint8)
    canvas3[:] = [255, 255, 255] # or img[:] = 255

    xs = []
    ys = []
    hs = []
    for b in boxes:
        xs.append(b[0])
        ys.append(b[1])
        hs.append(b[3])
    # iterations = math.ceil(float(max(ys)/min(ys)))
    # print (min(ys), max(ys), max(hs))
    # for iter in range(min(ys), max(ys), max(hs)):
    #     cv2.line(img, (0, iter), (w, iter), (0, 255, 0), 2)
    # x1 = max(hs)
    # ys1 = []
    # for y in ys:
    #     if y < (min(ys) + x1):
    #         ys1.append(y)



    top_line = [0,0,0,0]
    bottom_line = [0,0,0,0]
    for b in boxes:
        if b[0] == min(xs):
            top_line[0] = (b[0], int(b[1]*0.975))
            top_line[1] = (b[0], int((b[1]+b[3])*1.025))
        if b[1] == min(ys):
            top_line[2] = (b[0]+b[2], int(b[1]*0.95))
            top_line[3] = (b[0]+b[2], int((b[1]+b[3])*1))
        if b[0] == max(xs):
            bottom_line[0] = (b[0]+b[2], int(b[1]*0.975))
            bottom_line[1] = (b[0]+b[2], int((b[1]+b[3])*1.025))
        if b[1] == max(ys):
            bottom_line[2] = (b[0], b[1])
            bottom_line[3] = (b[0], b[1]+b[3])
    print top_line

    #cv2.rectangle(img, (top_line[0][0], top_line[0][1]), (top_line[3][0], top_line[3][1]), (0,0,255), 2)
    cv2.imshow('img', img)
    cv2.imwrite('../res/temp.png',img)
    cv2.waitKey(0)
    mask = np.zeros(img.shape, dtype=np.uint8)
    roi_corners = np.array([[(top_line[0]), (top_line[2]), (top_line[3]), (top_line[1])]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex

    sail_imgs = []
    # apply the mask
    img_mask = img[top_line[2][1]:top_line[1][1], top_line[0][0]:top_line[3][0]]
    mask = mask[top_line[2][1]:top_line[1][1], top_line[0][0]:top_line[3][0]]
    masked_image = cv2.bitwise_and(img_mask, mask)
    cv2.imwrite('../res/Sail Numbers/top.png', masked_image)
    sail_imgs.append(masked_image)

    mask = np.zeros(img.shape, dtype=np.uint8)
    roi_corners = np.array([[(bottom_line[0]), (bottom_line[2]), (bottom_line[3]), (bottom_line[1])]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex
    # apply the mask
    img_mask = img[bottom_line[0][1]:bottom_line[3][1], bottom_line[3][0]:bottom_line[0][0]]
    mask = mask[bottom_line[0][1]:bottom_line[3][1], bottom_line[3][0]:bottom_line[0][0]]
    masked_image = cv2.bitwise_and(img_mask, mask)
    cv2.imwrite('../res/Sail Numbers/bottom.png', masked_image)
    sail_imgs.append(masked_image)
    return sail_imgs



def get_sail_number(img):
    numbers = pd.read_csv('../res/sample sail numbers.csv', dtype={'ID': str})
    nums = [str(num[0]) for num in numbers.values]

    imgs = detect_digits(img)
    # for i, img in enumerate(imgs):
    #     cv2.imshow('{}'.format(i), img)
    # cv2.waitKey(0)

    imgs = four_point_transform(imgs, 0)
    # for i, img in enumerate(imgs):
    #     cv2.imshow('{}'.format(i), img)
    # cv2.waitKey(0)
    digits = []
    sail_num = []
    for i, img in enumerate(imgs):
        if i == 0:
            img = cv2.flip(img, 1)
        digits.append(recognise_digits(img))

    # for i in range(0,len(digits[0])):
    #         if digits[0][i] == digits[1][i]:
    #                 sail_num.append(digits[0][i])
    #         else:
    #                 sail_num.append('-')

    # print ''.join(sail_num)

    results = []
    #avg = math.ceil(float((len(digits[0]) + len(digits[1])))/2)
    avg = len(digits[0])

    #results.append(difflib.get_close_matches((''.join(sail_num)), nums))
    results.append(difflib.get_close_matches(digits[0], nums))
    #results.append(difflib.get_close_matches(digits[1], nums))
    print results
    # for result in results:
    #     for r in result:
    #         if len(r) == avg:
    #             print 'The sailing number is {}'.format(r)

img1 = cv2.imread('../res/sail_numbers_cropped.jpg')
#img1 = cv2.imread('../res/boat.png')
get_sail_number(img1)