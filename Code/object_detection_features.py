import numpy as np
import cv2
from matplotlib import pyplot as plt


img1 = cv2.imread('res/boat.jpg',0)          # queryImage
#img1 = cv2.resize(img1, (720,480) )
#cam = cv2.VideoCapture(0)
#cam = cv2.VideoCapture('res/KishRace6BoatCloseShort.mp4')
cam = cv2.VideoCapture('res/KishRace1.mp4')

while True:

    ret, img2 = cam.read()

    detector = cv2.SimpleBlobDetector_create()

    # Detect blobs.
    keypoints = detector.detect(img2)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(img2, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('blobs',im_with_keypoints)

    fast = cv2.FastFeatureDetector_create(80)



    # find and draw the key points
    kp = fast.detect(img2, None)
    pts = [p.pt for p in kp]


    img_out = cv2.drawKeypoints(img2, kp[:10],None, color=(255, 0, 0))
    cv2.circle(img_out, (int(pts[0][0]), int(pts[0][1])), 3, (0,255,0), thickness= 5)

    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    list_kp1 = []
    list_kp2 = []

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:29],None, flags=2)


    cv2.imshow('match', img3)
    cv2.imshow('original', img_out)

    cv2.waitKey(1)