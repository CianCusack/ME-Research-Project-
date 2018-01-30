import cv2
import imutils

# image = cv2.imread("../res/boat.jpg")
# grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(grey, 127, 255, 0)
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
#                         cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if imutils.is_cv2() else cnts[1]
# cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
# displayCnt = None
#
# # loop over the contours
# for c in cnts:
#     # approximate the contour
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#
#     # if the contour has four vertices, then we have found
#     # the thermostat display
#     if len(approx) == 4:
#         displayCnt = approx
#         break
# warped = four_point_transform(gray, displayCnt.reshape(4, 2))
# output = four_point_transform(image, displayCnt.reshape(4, 2))
# cv2.imshow("boat", thresh)
# cv2.waitKey(1000)

