import cv2
import numpy as np
file_name = "../res/test_data/rotated/7_rotated.png"

frame = cv2.imread(file_name)
frame = np.asarray(frame)
height, width, depth = frame.shape

for h in range(0,height):
    for w in range(0, width):
        for i, p in  enumerate(frame[h, w]):
            if p > 100:
                frame[h, w] = [255, 255, 255]
            else:
                frame[h, w] = [0, 0, 0]
#print frame
#frame = cv2.resize(frame,(400,400))
bordersize=5
dst=cv2.copyMakeBorder(frame, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255] )
cv2.imwrite("../res/test_data/final/7.png", dst)