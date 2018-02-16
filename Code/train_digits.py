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

#######   training part    ###############
samples = np.loadtxt('generalsamples.data', np.float32)
responses = np.loadtxt('generalresponses.data', np.float32)
responses = responses.reshape((responses.size, 1))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

############################# testing part  #########################

filenames = [glob.glob("../res/test_data/normal/*.png")]
filenames.sort()
filenames = [name.replace('"\"','/') for name in filenames[0]]
imgs = [cv2.imread(img) for img in filenames]
#imgs = [cv2.imread('../res/boat.jpg')]
strings = np.array([])
for img in imgs:
        out = np.zeros(img.shape, np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

        _,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        results = np.array([], dtype=np.int64)
        for cnt in contours:
                if cv2.contourArea(cnt) > 50:
                        [x, y, w, h] = cv2.boundingRect(cnt)
                        if h > 28:
                                #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                roi = thresh[y:y + h, x:x + w]
                                roismall = cv2.resize(roi, (10, 10))
                                roismall = roismall.reshape((1, 100))
                                roismall = np.float32(roismall)
                                retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)
                                value = int((results[0][0]))
                                string = str(int((results[0][0])))
                                #cv2.putText(img, string, (x+w/2, y + h/2), 0, 1, (0, 255, 0))
                                results = np.append(results, value)
        if len(results) > 0:
                results = results.astype(int)
                strings = np.append(strings, str(np.bincount(results).argmax()))


correct = [0,2,3,6,7,8]
images_and_labels = list(zip(imgs, correct))
for index, (image, label) in enumerate(images_and_labels):
    plt.subplot(2, 8, index+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Label: {}'.format(correct[index]))

# images_and_labels = list(zip(results, strings))
for index, (image) in enumerate(strings):
        out = np.zeros(img.shape, np.uint8)
        cv2.putText(out, strings[index], (x, y + h / 2), 0, 1, (0, 255, 0))
        plt.subplot(2, 8, index +9)
        plt.axis('off')
        plt.imshow(out, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Predicted {}'.format(strings[index]))

plt.show()
