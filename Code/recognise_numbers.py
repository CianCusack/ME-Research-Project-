import cv2
import numpy as np

def sort_contours(cnts, method="left-to-right"):

    # initialise the reverse flag and sort index
    reverse = False
    i = 0

    # If sorting in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # If sorting vertically
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    # Sort
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



    return ''.join(strings)

def guess_numbers(img):

    strings = np.array([])

    # create and train kNN model
    samples = np.loadtxt('generalsamples.data', np.float32)
    responses = np.loadtxt('generalresponses_slanted.data', np.float32)
    responses = responses.reshape((responses.size, 1))
    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # Within each individual image find the contours
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = np.array([])
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
                retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)

                value = int((results[0][0]))
                results = np.append(results, value)
                # If multiple numbers are found in image take number with greatest number of occurrences
            if len(results) > 0:

                results = results.astype(int)
                strings = np.append(strings, str(np.bincount(results).argmax()))

            else:

                strings = np.append(strings, '')
    return  "".join(strings)