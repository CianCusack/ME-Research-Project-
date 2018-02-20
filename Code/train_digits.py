# import sys
# import numpy as np
# import cv2
#
# im = cv2.imread('../res/training_image.png')
# im3 = im.copy()
#
# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (5, 5), 0)
# thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
#
# #################      Now finding Contours         ###################
#
# _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#
# samples = np.empty((0, 100))
# responses = []
# keys = [i for i in range(48, 58)]
#
# for cnt in contours:
#     if cv2.contourArea(cnt) > 50:
#         [x, y, w, h] = cv2.boundingRect(cnt)
#
#         if h  > 28:
#             cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
#             roi = thresh[y:y + h, x:x + w]
#             roismall = cv2.resize(roi, (10, 10))
#             cv2.imshow('norm', im)
#             key = cv2.waitKey(0)
#
#             if key == 27:  # (escape to quit)
#                 sys.exit()
#             elif key in keys:
#                 responses.append(int(chr(key)))
#                 sample = roismall.reshape((1, 100))
#                 samples = np.append(samples, sample, 0)
#
# responses = np.array(responses, np.float32)
# responses = responses.reshape((responses.size, 1))
# print "training complete"
# print samples
# np.savetxt('generalsamples.data',samples)
# np.savetxt('generalresponses.data',responses)


import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt

def recognise_digits(imgs):

        # train kNN model
        samples = np.loadtxt('generalsamples.data', np.float32)
        responses = np.loadtxt('generalresponses.data', np.float32)
        responses = responses.reshape((responses.size, 1))
        model = cv2.ml.KNearest_create()
        model.train(samples, cv2.ml.ROW_SAMPLE, responses)


        strings = np.array([])
        found_imgs = np.array([])
        for img in imgs:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

                _,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                results = np.array([])
                for cnt in contours:
                        if cv2.contourArea(cnt) > 50:
                                [x, y, w, h] = cv2.boundingRect(cnt)
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
        print strings
        # for index, image in enumerate(imgs):
        #         if(strings[index] == ''):
        #                 inc += 1
        #                 continue
        #         plt.subplot(2, len(strings[strings!=''])/2 +1 , index + 1 - inc)
        #         plt.axis('off')
        #         plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        #         plt.title('Predicted {}'.format(strings[index]))

        for index, image in enumerate(imgs):
                plt.subplot(2, len(strings)/2 +1 , index + 1 - inc)
                plt.axis('off')
                plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
                plt.title('Predicted {} '.format(strings[index]))
        plt.show()
        return strings


filenames = [glob.glob("../res/Sail Numbers/*.png")]
filenames.sort()
filenames = [name.replace('"\"','/') for name in filenames[0]]
imgs = [cv2.imread(img) for img in filenames]
#imgs = [cv2.resize(img, (30,30)) for img in imgs]

digits = recognise_digits(imgs[:10])
#print digits