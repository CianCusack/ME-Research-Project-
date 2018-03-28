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

