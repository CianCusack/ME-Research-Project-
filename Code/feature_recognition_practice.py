import cv2

def match_features(img1, img2):
    # Initiate SIFT detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with SIFT
    kp2, des2 = orb.detectAndCompute(img2,None)
    # Match descriptors.
    if len(kp2) != 0:
        return True
    else:
        return False

