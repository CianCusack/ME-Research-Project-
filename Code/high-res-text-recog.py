import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
from sklearn import datasets, svm

digits = datasets.load_digits()

images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:9]):
    plt.subplot(3, 9, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

classifier = svm.SVC(gamma=0.001)

classifier.fit(data[:n_samples], digits.target[:n_samples])

# Now predict the value of the digit on the second half:
#img = cv2.imread('../res/Sail Numbers/92.png')

filenames = [glob.glob("../res/test_data/*.png")]
filenames.sort()
filenames = [name.replace('"\"','/') for name in filenames[0]]
imgs = [cv2.imread(img) for img in filenames]
predictions = []
found_imgs = []
missed = []
missed_imgs = []
for img in imgs:
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, (8, 8))
    img = img[:, :, 0]

    minValueInImage = np.min(img)
    maxValueInImage = np.max(img)
    normalizeImg = np.floor(np.divide((img - minValueInImage).astype(np.float),(maxValueInImage-minValueInImage).astype(np.float))*16)
    normalizeImg = normalizeImg.reshape((1,normalizeImg.shape[0]*normalizeImg.shape[1] ))

    predicted = classifier.predict(normalizeImg)
    expected = digits.target[1]
    print predicted
    if predicted != 1:
        found_imgs.append(img)
        predictions.append(predicted)
    else:
        missed.append(predicted)
        missed_imgs.append(img)

images_and_labels = list(zip(found_imgs, predictions))
for index, (image, label) in enumerate(images_and_labels):
    plt.subplot(3, 9, index + 10)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: {}'.format(predictions[index]))

images_and_labels = list(zip(missed_imgs, missed))
for index, (image, label) in enumerate(images_and_labels):
    plt.subplot(3, 9, index + 19)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Missed {}'.format(missed[index]))



plt.show()




