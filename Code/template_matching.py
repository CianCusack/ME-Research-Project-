import cv2
from feature_recognition_practice import *


def match_template(img, template, bounds):
    w, h = template.shape[:2]
    img2 = img.copy()


    res = cv2.matchTemplate(img2,template, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    bottom_right = (min_loc[0] + w, min_loc[1] + h)
    x1, y1, x2, y2 =min_loc[0], min_loc[1], bottom_right[0], bottom_right[1]

    # hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv, bounds[0], bounds[1])
    # res_template = cv2.bitwise_and(template, template, mask=mask)
    #
    # if y1-50 >0 and x1-50 >0 and x2+50 < 1280 and y2+50 < 720:
    #     img3 = img2[y1 - 50:y2 + 50, x1 - 50:x2 + 50].copy()
    #     hsv = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
    #     mask = cv2.inRange(hsv, bounds[0], bounds[1])
    # else:
    #     img3 = img2.copy()
    #     hsv = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
    #     mask = cv2.inRange(hsv, bounds[0], bounds[1])
    #
    # res_img = cv2.bitwise_and(img3, img3, mask=mask)
    # if match_features(res_template, res_img):
    return x1,y1,x2,y2
    # else:
    #     return 0,0,w,h