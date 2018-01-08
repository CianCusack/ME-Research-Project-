import urllib
import cv2
import numpy as np
import os


def get_pos():
    urls = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n04127904'
    neg_image_urls = urllib.urlopen(urls).read().decode()

    pic_num = 1

    if not os.path.exists('pos'):
        os.makedirs('pos')

    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.urlretrieve(i, "pos/" + str(pic_num) + ".jpg")
            img = cv2.imread("pos/" + str(pic_num) + ".jpg", cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite("pos/" + str(pic_num) + ".jpg", resized_image)
            pic_num += 1

        except Exception as e:
            print(str(e))

def get_neg():
    #urls = "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n09428293"
    urls = "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n09436708"
    neg_image_urls = urllib.urlopen(urls).read().decode()
    pic_num = 777

    if not os.path.exists('neg'):
        os.makedirs('neg')

    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.urlretrieve(i, "neg/" + str(pic_num) + ".jpg")
            img = cv2.imread("neg/" + str(pic_num) + ".jpg", cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite("neg/" + str(pic_num) + ".jpg", resized_image)
            pic_num += 1

        except Exception as e:
            print(str(e))



def find_uglies():
    match = False
    for file_type in ['neg']:
        for img in os.listdir(file_type):
            for ugly in os.listdir('uglies'):
                try:
                    current_image_path = str(file_type) + '/' + str(img)
                    ugly = cv2.imread('uglies/' + str(ugly))
                    question = cv2.imread(current_image_path)
                    if ugly.shape == question.shape and not (np.bitwise_xor(ugly, question).any()):
                        print('That is one ugly pic! Deleting!')
                        print(current_image_path)
                        os.remove(current_image_path)
                except Exception as e:
                    print(str(e))

def resize_all():
    for file_type in ['neg']:
        for img in os.listdir(file_type):
            try:
                current_image_path = str(file_type) + '/' + str(img)
                img = cv2.imread(current_image_path)
                img = cv2.resize(img, (200,200))
                cv2.imwrite(current_image_path, img)
            except Exception as e:
                print(str(e))

"""def store_raw_images():
    neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n04127904'
    neg_image_urls = urllib.urlopen(neg_images_link).read().decode()
    pic_num = 869

    if not os.path.exists('neg'):
        os.makedirs('neg')

    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.urlretrieve(i, "neg/" + str(pic_num) + ".jpg")
            img = cv2.imread("neg/" + str(pic_num) + ".jpg", cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite("neg/" + str(pic_num) + ".jpg", resized_image)
            pic_num += 1

        except Exception as e:
            print(str(e))"""


def create_neg():
    for file_type in ['neg']:

        for img in os.listdir(file_type):

            if file_type == 'pos':
                line = file_type + '/' + img + '\n'
                with open('info.dat', 'a') as f:
                    f.write(line)
            elif file_type == 'neg':
                #line = file_type + '/' + img + '\n'
                line = '../'+file_type + '/' + img + '\n'
                line = line.replace('"\\"', '"\"')
                with open('negatives.txt', 'a') as f:
                    f.write(line)

def create_pos():
    for file_type in ['pos']:

        for img in os.listdir(file_type):

            if file_type == 'pos':
                #line = file_type + '/' + img + '\n'
                line = '../' + file_type + '/' + img + ' 1 0 0 200 200\n'
                line = line.replace('"\\"', '"\"')
                with open('positive.txt', 'a') as f:
                    f.write(line)

def use_cascade(img):
    boat_cascade = cv2.CascadeClassifier('../bin/data/cascade.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    boats = boat_cascade.detectMultiScale(gray, scaleFactor = 1.01, minNeighbors = 3)
    return boats



