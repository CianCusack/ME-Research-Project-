import cv2

def get_extreme_point(img):
    h, w = img.shape[:2]
    #img = img[(2*h)/3 : h, 0 : w]
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray.copy(), 128, 255, cv2.THRESH_BINARY_INV)
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                              cv2.THRESH_BINARY_INV)

    temp_x = []
    temp_y = []
    for c in cnts:
        for c1 in c:
            # cv2.drawContours(img, c, -1, (255,0,0), thickness=2)
            temp_x.append(c1[0][0])
            temp_y.append(c1[0][1])
    if len(cnts) == 0:
        return 0,0

    extreme_points = []
    for x,y in zip(temp_x, temp_y):
        thresh = (2.0/3.0)*float(h)
        if y < thresh:
            temp_x.remove(x)
            temp_y.remove(y)
    for index, x in enumerate(list(temp_x)):
        if x == max(temp_x):
            extreme_points.append([ x, temp_y[index]])
    cv2.circle(img, (max(extreme_points)[0], max(extreme_points)[1]), 2, (0,0,255), 3)
    return max(extreme_points)[0], (min(extreme_points)[1])


# img = cv2.imread('../res/test/test100.png')
# get_extreme_point(img)