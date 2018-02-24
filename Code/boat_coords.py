import cv2
import matplotlib.pyplot as plt
def get_extreme_point(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray.copy(), 127, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('thresh', thresh)
    # cv2.waitKey(0)
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                              cv2.THRESH_BINARY_INV)
    # ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # ret, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    # ret, thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
    # ret, thresh4 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
    # ret, thresh5 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)
    # titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    # images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    # for i in xrange(6):
    #     plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    # plt.show()
    temp_x = []
    temp_y = []
    for c in cnts:
        for c1 in c:
            #cv2.drawContours(img, c, -1, (255,0,0), thickness=2)
            temp_x.append(c1[0][0])
            temp_y.append(c1[0][1])
    if len(cnts) == 0:
        return 0,0

    points = []
    for x,y in zip(temp_x, temp_y):
        thresh = (2.0/3.0)*float(h)
        if y < thresh or x < max(temp_x)*0.9:
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