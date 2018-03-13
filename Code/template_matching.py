import cv2


def match_template(img, template):
    w, h = template.shape[:2]
    res = cv2.matchTemplate(img.copy(),template, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    bottom_right = (min_loc[0] + w, min_loc[1] + h)
    return min_loc[0], min_loc[1], bottom_right[0], bottom_right[1]
