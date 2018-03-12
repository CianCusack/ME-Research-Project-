import cv2
def get_extreme_point(img):
    h, w = img.shape[:2]
    if h < 10 or w < 10:
        return None
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray.copy(), 127, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('thresh', thresh)
    # cv2.waitKey(0)
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                              cv2.THRESH_BINARY_INV)
    temp_x = []
    temp_y = []
    thresh = (2.0/3.0)*float(h)
    flat_list = [internal_item for sublist in cnts for item in sublist for internal_item in item]
    for f in flat_list:
        if f[1] > thresh:
            temp_x.append(f[0])
            temp_y.append(f[1])
    if len(cnts) == 0 or len(temp_x) == 0:
        return None

    points = []
    max_x = max(temp_x)
    for x,y in zip(temp_x, temp_y):
        if x < max_x*0.9:
            continue
        points.append((x, y))

    sorted_by_x = sorted(points, key=lambda tup: tup[0])
    sorted_by_x = sorted_by_x[::-1]
    # for p in points:
    #     cv2.circle(img, (p), 2, (0,0,255), 3)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    return sorted_by_x[:10]


img = cv2.imread('../res/test/test100.png')
get_extreme_point(img)