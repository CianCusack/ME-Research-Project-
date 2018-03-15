import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
from sklearn import datasets, svm
import idx2numpy
import pickle

size = 16

#Train with sklearn
# digits = datasets.load_digits()
# images = np.array([cv2.resize(img, (size,size)) for img in digits.images])
# images_and_labels = list(zip(images, digits.target))
# for index, (image, label) in enumerate(images_and_labels[:9]):
#     plt.subplot(3, 9, index + 1)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Training: %i' % label)
#
# n_samples = len(images)
# data = images.reshape((n_samples, -1))
# classifier = svm.SVC(gamma=0.001)
# classifier.fit(data, digits.target)

#Train with mnist
#
mnist_data = idx2numpy.convert_from_file('../res/mnist/train-images.idx3-ubyte')
mnist_labels = idx2numpy.convert_from_file('../res/mnist/train-labels.idx1-ubyte')
imgs = []
for img in mnist_data:
    img = cv2.resize(img, (8,8))
    #img = img[:,:,0]
    minValueInImage = np.min(img)
    maxValueInImage = np.max(img)
    normaliseImg = np.floor(np.divide((img - minValueInImage).astype(np.float),(maxValueInImage-minValueInImage).astype(np.float))*16)
    normaliseImg = normaliseImg.reshape((1,normaliseImg.shape[0]*normaliseImg.shape[1] ))
    imgs.append(img)

images_and_labels = list(zip(mnist_data, mnist_labels))
for index, (image, label) in enumerate(images_and_labels[:9]):
    plt.subplot(3, 9, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

print '************** Starting Training **************'
# n_samples = len(mnist_data)
# data = mnist_data.reshape((n_samples, -1))
# classifier = svm.SVC(gamma=(1/float(n_samples)))
# classifier.fit(data, mnist_labels)
#
# pkl_filename = "digits.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(classifier, file)


classifier = pickle.load(open('digit.pkl', 'rb'))
print '************** Finished Training **************'
#Predict
filenames = [glob.glob("../res/test_data/final/*.png")]
filenames.sort()
filenames = [name.replace('"\"','/') for name in filenames[0]]
imgs = [cv2.imread(img) for img in filenames]
#imgs = [cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)[1] for img in imgs]
predictions = []
found_imgs = []
missed = []
missed_imgs = []
for img in imgs:
    img = cv2.bitwise_not(img)
    #img = cv2.GaussianBlur(img,(5,5),0)
    img = cv2.resize(img, (size, size))
    img = img[:, :, 0]

    minValueInImage = np.min(img)
    maxValueInImage = np.max(img)
    normalizeImg = np.floor(np.divide((img - minValueInImage).astype(np.float),(maxValueInImage-minValueInImage).astype(np.float))*16)
    normalizeImg = normalizeImg.reshape((1,normalizeImg.shape[0]*normalizeImg.shape[1] ))
    normalizeImg = img.reshape((1, img.shape[0] * img.shape[1]))
    predicted = classifier.predict(normalizeImg)

    print predicted
    if predicted != 1:
        found_imgs.append(img)
        predictions.append(predicted)
    else:
        missed.append(predicted)
        missed_imgs.append(img)

images_and_labels = list(zip(found_imgs, predictions))
for index, (image, label) in enumerate(images_and_labels):
    plt.subplot(3, 9, index+10)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: {}'.format(predictions[index]))

images_and_labels = list(zip(missed_imgs, missed))
for index, (image, label) in enumerate(images_and_labels[:9]):
    plt.subplot(3, 9, index +19)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Missed {}'.format(missed[index]))



plt.show()




