
import cv2
import numpy as np
from matplotlib import pyplot as plt


def sort_contours(cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0

        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
                reverse = True

        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
                i = 1

        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))

        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)

def recognise_digits(img):

        # train kNN model
        samples = np.loadtxt('generalsamples.data', np.float32)
        responses = np.loadtxt('generalresponses_slanted.data', np.float32)
        responses = responses.reshape((responses.size, 1))
        model = cv2.ml.KNearest_create()
        model.train(samples, cv2.ml.ROW_SAMPLE, responses)


        strings = np.array([])
        images = []

        h, w = img.shape[:2]

        img = cv2.resize(img, (w*2, h*2))
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray.copy(), 127, 255, cv2.THRESH_BINARY_INV)
        _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                          cv2.THRESH_BINARY_INV)
        contours , _ = sort_contours(contours)
        for c in contours:
                if cv2.contourArea(c) < 300:
                        continue

                temp_x = []
                temp_y = []
                for c1 in c:
                        temp_x.append(c1[0][0])
                        temp_y.append(c1[0][1])

                images.append(img[min(temp_y):max(temp_y), min(temp_x): max(temp_x)].copy())
        for img in images:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

                _,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                results = np.array([])
                for cnt in contours:
                        if cv2.contourArea(cnt) > 40:
                                [x, y, w, h] = cv2.boundingRect(cnt)
                                # cv2.drawContours(img, cnt, -1, (255,0,0), 2)
                                # cv2.imshow('img', img)
                                # cv2.waitKey(0)
                                if h > 28:
                                        roi = thresh[y:y + h, x:x + w]
                                        roismall = cv2.resize(roi, (10, 10))
                                        roismall = roismall.reshape((1, 100))
                                        roismall = np.float32(roismall)
                                        retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)
                                        value = int((results[0][0]))
                                        results = np.append(results, value)
                # If multiple numbers are found in image take number with greatest number of occurrences

                if len(results) > 0:
                        results = results.astype(int)
                        strings = np.append(strings, str(np.bincount(results).argmax()))
                else:
                        strings = np.append(strings, '')
        inc = 0
        #print strings
        for index, image in enumerate(images):
                if(strings[index] == ''):
                        inc += 1
                        continue
                plt.subplot(2, len(strings[strings!=''])/2 +1 , index + 1 - inc)
                plt.axis('off')
                plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
                plt.title('Predicted {}'.format(strings[index]))

        for index, image in enumerate(images):
                plt.subplot(2, len(strings)/2 +1 , index + 1 - inc)
                plt.axis('off')
                plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
                plt.title('Predicted {} '.format(strings[index]))
        plt.show()
        return ''.join(strings)

