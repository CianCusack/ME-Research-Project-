import cv2


def match_template(img, template, last_location, distance):
    h, w = template.shape[:2]
    if sum(last_location) != 0 and distance > 80:
        f = lambda a: (abs(a) + a) / 2
        y1, y2, x1, x2 = int(last_location[1] - (150)), int(last_location[3] + (150)), int(last_location[0] - (150)), int(last_location[2] + (150))
        y1, y2, x1, x2 = f(y1), f(y2), f(x1), f(x2)
        img1 = img[y1: y2, x1: x2]
        if img1.shape[0] < w or img1.shape[1] < h:
            res = cv2.matchTemplate(img.copy(), template, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = (min_loc[0], min_loc[1])
            bottom_right = (top_left[0] + w, top_left[1] + h)
        else:
            res = cv2.matchTemplate(img1.copy(), template, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = (min_loc[0] + int(last_location[0] - (2*w)), min_loc[1] + int(last_location[1] - (2*h)))
            bottom_right = (top_left[0] + w, top_left[1] + h)
    else:
        res = cv2.matchTemplate(img.copy(), template, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = (min_loc[0], min_loc[1])
        bottom_right = (top_left[0] + w, top_left[1] + h)

    return top_left[0], top_left[1], bottom_right[0], bottom_right[1]
