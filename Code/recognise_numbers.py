import cv2
import numpy as np
from imutils import rotate

def sort_contours(cnts, method="left-to-right"):

    # initialise the reverse flag and sort index
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
                                        key=lambda b: b[1][0], reverse=reverse))
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def recognise_digits(img):




    images = []
    strings = []
    # Increase image size so edges are easier to detect
    h, w = img.shape[:2]
    if h < 30:

        img = cv2.resize(img, (w*3, h*3))

    else:

        img = cv2.resize(img, (w * 2, h * 2))

    # Find contours in the frame
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray.copy(), 127, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                      cv2.THRESH_BINARY_INV)

    # If no contours, no numbers recognised
    if len(contours) == 0:
        return ''

    contours , _ = sort_contours(contours)

    for c in contours:
        # Ignore small contours
        if cv2.contourArea(c) < 300:
            continue

        temp_x = []
        temp_y = []
        for c1 in c:
            temp_x.append(c1[0][0])
            temp_y.append(c1[0][1])

        #Store image in the contour areas
        images.append(img[min(temp_y):max(temp_y), min(temp_x): max(temp_x)].copy())

    for img in images:

        strings.append(guess_numbers(img))

    return strings

def guess_numbers(img):

    strings = np.array([])
    strings_rotated = np.array([])
    strings_rotated_mirrored = np.array([])

    # create and train kNN model
    # samples = np.loadtxt('generalsamples.data', np.float32)
    # responses = np.loadtxt('generalresponses_slanted.data', np.float32)
    samples = np.loadtxt('redesign_samples1.data', np.float32)
    responses = np.loadtxt('redesign_responses1.data', np.float32)
    responses = responses.reshape((responses.size, 1))
    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # Within each individual image find the contours
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = np.array([])
    results_rotated = np.array([])
    results_rotated_mirrored = np.array([])

    for cnt in contours:
        if cv2.contourArea(cnt) > 40:
            # cv2.drawContours(img, cnt, -1, (0,0,255), 2)
            # cv2.imshow('detection', img)
            # cv2.waitKey(0)
            [x, y, w, h] = cv2.boundingRect(cnt)

            if h > 28:
                roi = thresh[y:y + h, x:x + w]
                roismall = cv2.resize(roi, (10, 10))
                roismall = roismall.reshape((1, 100))
                roismall = np.float32(roismall)

                # Use kNN model to try identify digit
                value, result, neigh_resp, dists = model.findNearest(roismall, k=11)
                cv2.imshow('roismall', roi)
                print 'Original {}, {}, {}'.format(value, result, neigh_resp)

                roi = rotate(roi, -15)
                roismall = cv2.resize(roi, (10, 10))
                roismall = roismall.reshape((1, 100))
                roismall = np.float32(roismall)

                # Use kNN model to try identify digit
                value_rotated, result, neigh_resp, dists = model.findNearest(roismall, k=11)
                cv2.imshow('roismall rotated', roi)
                print 'Rotated {}, {}, {}'.format(value_rotated, result, neigh_resp)

                roi = cv2.flip(roi, 1)
                roismall = cv2.resize(roi, (10, 10))
                roismall = roismall.reshape((1, 100))
                roismall = np.float32(roismall)

                # Use kNN model to try identify digit
                value_rotated_mirrored, result, neigh_resp, dists = model.findNearest(roismall, k=3)
                print 'Rotated and mirrored {}, {}, {}'.format(value_rotated_mirrored, result, neigh_resp)

                cv2.imshow('roismall rotated, mirrored', roi)
                cv2.waitKey(1)
                cv2.destroyAllWindows()

                results = np.append(results, value)
                results_rotated = np.append(results_rotated, value_rotated)
                results_rotated_mirrored = np.append(results_rotated_mirrored, value_rotated_mirrored)
                # If multiple numbers are found in image take number with greatest number of occurrences
            if len(results) > 0:

                results = results.astype(int)
                results_rotated = results_rotated.astype(int)
                results_rotated_mirrored = results_rotated_mirrored.astype(int)
                strings = np.append(strings, str(np.bincount(results).argmax()))
                strings_rotated = np.append(strings_rotated, str(np.bincount(results_rotated).argmax()))
                strings_rotated_mirrored = np.append(strings_rotated_mirrored, str(np.bincount(results_rotated_mirrored).argmax()))

            else:

                strings = np.append(strings, '')
    x= ["".join(strings), "".join(strings_rotated), "".join(strings_rotated_mirrored)]
    y = [results, results_rotated, results_rotated_mirrored]
    print x, y
    return  x