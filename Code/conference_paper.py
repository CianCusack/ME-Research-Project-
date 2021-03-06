# from boat_detector import *
#
# img = cv2.imread('../res/Hi-Res Boats/boat11.jpg')
# # img = cv2.imread('../res/boats/4.png')
# # img = cv2.imread('../res/test/test1.png')
# h, w = img.shape[:2]
#
# # Draw original image
# _, coords = detect_boats(img)
# for c in coords:
#     cv2.rectangle(img , (c[0], c[1]), (c[2], c[3]), (0,0,255), 2)
#     print c[2] - c[0], c[3] - c[1]
# cv2.imshow('img', img)
# #cv2.imwrite('../../Conference Paper/images/boat_detection.png', img)
#
# # Draw resize images
# for i in range(1, 8):
#     img = cv2.imread('../res/Hi-Res Boats/boat11.jpg')
#     # img = cv2.imread('../res/boats/4.png')
#     # img = cv2.imread('../res/test/test1.png')
#     img = cv2.resize(img, (w/(2**i), h/(2**i)))
#     print img.shape[:2]
#
#     _, coords = detect_boats(img)
#
#     for c in coords:
#         cv2.rectangle(img , (c[0], c[1]), (c[2], c[3]), (0,0,255), 2)
#
#     cv2.imshow('img {}'.format(i), img)
# cv2.waitKey(0)
import numpy as np
import cv2

# samples = np.loadtxt('redesign_samples1.data', np.float32)
# responses = np.loadtxt('redesign_responses1.data', np.float32)
# responses = responses.reshape((responses.size, 1))
# model = cv2.ml.KNearest_create()
# model.train(samples, cv2.ml.ROW_SAMPLE, responses)
#
#
# im = cv2.imread('../res/sail_nums_example.png')
# gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
#
#
# _, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#
# for cnt in contours:
#     if cv2.contourArea(cnt) > 40 :
#         [x, y, w1, h1] = cv2.boundingRect(cnt)
#         if h1 > 50:
#             roi = thresh[y:y + h1, x:x + w1]
#             roismall = cv2.resize(roi, (10, 10))
#             roismall = roismall.reshape((1, 100))
#             roismall = np.float32(roismall)
#             retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)
#             string = str(int((results[0][0])))
#             #result.append(string)
#             cv2.rectangle(im, (x, y), (x + w1, y + h1), (0, 0, 255), 2)
#             cv2.putText(im, string, (x+w1/2, y -5), 0, 1, (0, 255, 0), 2)
# cv2.imshow('img', im)
# cv2.imwrite('../../conference paper/images/digit_recognition.png', im)
# cv2.waitKey(0)

from digit_recognition_new import *
for i in range(1, 25):
    img = cv2.imread('../res/boats/{}.png'.format(i))
    print detect_sail_number(img)