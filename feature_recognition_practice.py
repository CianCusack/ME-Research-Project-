import numpy as np
import cv2
from matplotlib import pyplot as plt

def match_features(img1, img2):
    x_max = 0
    x_min = 1280
    y_max = 0
    y_min = 720
    # Initiate SIFT detector
    orb = cv2.ORB_create()


    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    if len(kp1) != 0 and len(kp2) != 0:

        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        img4 = cv2.drawKeypoints(img2, kp2, None, (255,255,255), flags = 0)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

        for p in kp2:
            if p.pt[0] > x_max:
                x_max = p.pt[0]
            if p.pt[0] < x_min:
                x_min = p.pt[0]
            if p.pt[1] > y_max:
                y_max = p.pt[1]
            if p.pt[1] < y_min:
                y_min = p.pt[1]


        cv2.rectangle(img4, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,255,0))
        #plt.imshow(img3), plt.show()
        #cv2.imshow('img4', img4)
        #cv2.waitKey(1)
    return int(x_min), int(y_min), int(x_max), int(y_max)


