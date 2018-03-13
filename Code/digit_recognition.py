import numpy as np
import cv2
from four_point_transform import *
from recognise_numbers import *
import difflib
import pandas as pd
import math

def get_sail_number_line(boxes, img):
    xs = []
    ys = []
    hs = []
    for b in boxes:
        xs.append(b[0])
        ys.append(b[1])
        hs.append(b[3])

    top_line = [0, 0, 0, 0]
    bottom_line = [0, 0, 0, 0]
    for b in boxes:
        if b[0] == min(xs):
            top_line[0] = (b[0], int(b[1] * 0.975))
            top_line[1] = (b[0], int((b[1] + b[3]) * 1.025))
        if b[1] == min(ys):
            top_line[2] = (b[0] + b[2], int(b[1] * 0.95))
            top_line[3] = (b[0] + b[2], int((b[1] + b[3]) * 1))
        if b[0] == max(xs):
            bottom_line[0] = (b[0] + b[2], int(b[1] * 0.975))
            bottom_line[1] = (b[0] + b[2], int((b[1] + b[3]) * 1.025))
        if b[1] == max(ys):
            bottom_line[2] = (b[0], b[1])
            bottom_line[3] = (b[0], b[1] + b[3])

    sail_imgs = []
    mask = np.zeros(img.shape, dtype=np.uint8)
    roi_corners = np.array([[(top_line[0]), (top_line[2]), (top_line[3]), (top_line[1])]], dtype=np.int32)
    ignore_mask_color = (255,) * img.shape[2]
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)

    img_mask = img[top_line[2][1]:top_line[1][1], top_line[0][0]:top_line[3][0]]
    mask = mask[top_line[2][1]:top_line[1][1], top_line[0][0]:top_line[3][0]]
    masked_image = cv2.bitwise_and(img_mask, mask)
    cv2.imwrite('../res/Sail Numbers/top.png', masked_image)
    sail_imgs.append(masked_image)

    mask = np.zeros(img.shape, dtype=np.uint8)
    roi_corners = np.array([[(bottom_line[0]), (bottom_line[2]), (bottom_line[3]), (bottom_line[1])]], dtype=np.int32)
    ignore_mask_color = (255,) * img.shape[2]
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)

    img_mask = img[bottom_line[0][1]:bottom_line[3][1], bottom_line[3][0]:bottom_line[0][0]]
    mask = mask[bottom_line[0][1]:bottom_line[3][1], bottom_line[3][0]:bottom_line[0][0]]
    masked_image = cv2.bitwise_and(img_mask, mask)
    cv2.imwrite('../res/Sail Numbers/bottom.png', masked_image)
    sail_imgs.append(masked_image)

    return sail_imgs

def detect_digits(img):
    h, w = img.shape[:2]
    if h * w < 200:
        return None, -1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get mser, and set parameters
    mser = cv2.MSER_create()
    mser.setMinArea(60)
    mser.setMaxArea(200)

    # Do mser detection, get the coodinates and bboxes
    coordinates, bboxes = mser.detectRegions(gray)

    # Filter the coordinates
    if len(coordinates) == 0:
        return None, -1

    coordinates, _ = sort_contours(coordinates, 'top-to-bottom')
    coords = []
    boxes = []
    angles = []
    i = 0
    for coord in coordinates:
        bbox = cv2.boundingRect(coord)
        x,y,w1,h1 = bbox
        if w1< 8 or h1 < 15 or w1 > h1 or h1/w1 > 10:
            continue

        coords.append(coord)
        boxes.append(bbox)
        rect = cv2.minAreaRect(coord)
        theta =  abs(rect[2]/45)
        if theta == 0 or theta == 1:
            continue
        angles.append(theta)

    if len(angles) == 0:
        return None, -1

    average_angle =  sum(angles)/len(angles)
    canvas3 = np.zeros([img.shape[0],img.shape[1],3],dtype=np.uint8)
    canvas3[:] = [255, 255, 255] # or img[:] = 255
    mode = 0

    # Determine if the angle of the sail numbers is small enough to ignore
    if 2 - average_angle < 0.8 or 2-average_angle > 1.2:
        sail_imgs = get_sail_number_line(boxes, img)
        mode = 1
    else:
        xs = []
        ys = []
        hs = []
        ws = []
        for b in boxes:
            xs.append(b[0])
            ys.append(b[1])
            ws.append(b[2])
            hs.append(b[3])

        top_xs= []
        top_ys = []
        for b in boxes:
            if b[1] > min(ys) + max(hs):
                    continue
            top_xs.append(b[0])
            top_ys.append(b[1])
        sail_imgs = [img[min(top_ys): max(top_ys)+max(hs), min(top_xs):max(top_xs)+max(ws)].copy()]

    return sail_imgs, mode



def get_sail_number(img):
    numbers = pd.read_csv('../res/sample sail numbers.csv', dtype={'ID': str})
    nums = [str(num[0]) for num in numbers.values]

    imgs, mode = detect_digits(img)
    if mode == 1:
        imgs = four_point_transform(imgs, 0)
    if mode == -1 or imgs == None:
        return
    digits = []
    for i, img in enumerate(imgs):
        if i == 0 and mode == 1:
            img = cv2.flip(img, 1)
        digits.append(recognise_digits(img))
    print digits

    results = []

    results.append(difflib.get_close_matches(digits[0], nums))
    if len(results) == 0:
        return 0

    return results[0]
