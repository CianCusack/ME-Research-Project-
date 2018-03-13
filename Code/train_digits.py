import sys
import numpy as np
import cv2
import glob

imgs = []
for img in glob.glob("../res/training images/*.jpg"):
    imgs.append(cv2.imread(img))
samples = np.empty((0, 100))
responses = []
for im in imgs:

        im3 = im.copy()

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

        #################      Now finding Contours         ###################

        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        keys = [i for i in range(48, 58)]

        for cnt in contours:
            if cv2.contourArea(cnt) > 40:
                [x, y, w, h] = cv2.boundingRect(cnt)
                if h  > 28:
                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    roi = thresh[y:y + h, x:x + w]
                    roismall = cv2.resize(roi, (10, 10))
                    cv2.imshow('norm', im)
                    key = cv2.waitKey(0)

                    if key == 27:  # (escape to quit)
                        sys.exit()
                    elif key in keys:
                        responses.append(int(chr(key)))
                        sample = roismall.reshape((1, 100))
                        samples = np.append(samples, sample, 0)


responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))
print "training complete"
print len(samples)
print responses
np.savetxt('generalsamples_1.data',samples)
np.savetxt('generalresponses_slanted_1.data',responses)

