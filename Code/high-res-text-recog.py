# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, preprocessing

# The digits dataset
digits = datasets.load_digits()

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples], digits.target[:n_samples])

# Now predict the value of the digit on the second half:
img = cv2.imread('../res/Sail Numbers/92.png')
img = cv2.bitwise_not(img)
img = cv2.flip(img, 1)
img = cv2.resize(img, (8, 8))
img = img[:, :, 0]

# img = img.reshape(1,img.shape[0]*img.shape[1])
# img = img.astype(np.uint8)
minValueInImage = np.min(img)
maxValueInImage = np.max(img)
normaliizeImg = np.floor(np.divide((img - minValueInImage).astype(np.float),(maxValueInImage-minValueInImage).astype(np.float))*16)
normaliizeImg = normaliizeImg.reshape((1,normaliizeImg.shape[0]*normaliizeImg.shape[1] ))
# Predict
predicted = classifier.predict(normaliizeImg)


expected = digits.target[1]
print predicted
# print("Classification report for classifier %s:\n%s\n"
#       % (classifier, metrics.classification_report(expected, predicted)))
# print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


plt.subplot(2, 4, 5)
plt.axis('off')
plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
plt.title('Prediction: {}, Expected: {}'.format(predicted[0], expected))

plt.show()