import cv2

def get_extreme_point(img):
    #img = cv2.resize(img, (400,400))
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray.copy(), 75, 128, cv2.THRESH_BINARY_INV)
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                              cv2.THRESH_BINARY_INV)

    temp_x = []
    temp_y = []
    for c in cnts:
        #cv2.drawContours(img, c, -1, (0,0,0), thickness=2)
        for c1 in c:
            temp_x.append(c1[0][0])
            temp_y.append(c1[0][1])
    if len(cnts) == 0:
        return 0,0
    max_x = max(temp_x)

    extreme_points = []
    for index, x in enumerate(list(temp_x)):
        if x == max_x:
            extreme_points.append([ x, temp_y[index]])
    return max(extreme_points)[0], max(extreme_points)[1]


# img = cv2.imread('../res/test/test100.png')
# get_extreme_point(img)