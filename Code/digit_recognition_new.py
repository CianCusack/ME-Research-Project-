# import packages
import cv2
import numpy as np
import difflib
import pandas as pd

# This is all done at before the program starts so that it is only done once
# rather than each time the method is called.

# Read in trained data set and labels
samples = np.loadtxt('redesign_samples1.data', np.float32)
responses = np.loadtxt('redesign_responses1.data', np.float32)
responses = responses.reshape((responses.size, 1))

# Train KNN model using data set and labels
model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

# Read in known sail numbers
numbers = pd.read_csv('../res/sample sail numbers.csv', dtype={'ID': str})
nums = [str(num[0]) for num in numbers.values]

# Returns most likely value of sail number in a image
def detect_sail_number(im):

    # Convert image to gray scale and threshold
    h, w = im.shape[:2]
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)


    # Do MSER detection, get the coordinates and bounding boxes of possible text areas
    mser = cv2.MSER_create()
    coordinates, bounding_boxes = mser.detectRegions(gray)

    # Sort bounding boxes from top to bottom
    bounding_boxes = sorted(bounding_boxes, key=lambda k: [k[1], k[0]])

    last_x, last_y, last_h = w, h, 0
    results_location = []

    for cnt in bounding_boxes:

        # Extract x,y position and width and height
        x,y,w1,h1 = cnt

        # Ignore regions below 50 pixels in area
        if w1*h1>50:

            # If the region is greater than 10% of the width or height ignore
            if float(h1)/float(h) > (0.1) or float(w1)/float(w) > (0.1):
                continue

            # If the region has already been detected skip
            if last_x == x and last_y ==y:
                continue

            # If the region is in the bottom 1/3 ignore as most likely just boat
            if y > (4.0/6.0)*h:
                continue

            # Ignore any regions less than pixels in height as they will not accurately be detected
            if h1 > 10:

                # Update with new values
                last_x, last_y, last_h = x, y, h1
                results_location.append((x,y,w1,h1))

    last_x, last_y, last_h = w, h, 0
    row_height = 0
    rows = []
    row = []

    for p in results_location:

        (x,y,w1,h1) = p

        # Ignore areas very near to already detected regions
        if h1 < 0.9*row_height and abs(y-last_y) < 0.9*row_height:
            continue

        # Determine rows with significant x and y change
        if abs(y-last_y) > 0 and abs(x-last_x) > 2*w1:
            rows.append(row)
            row = []
            row_height = h1
        row.append((x,y,w1,h1))
        last_x, last_y, last_h = x, y, h1

    sail_nums = []
    for r in rows:

        sail_num = []
        last_x = w

        # Sort rows from left to right
        r = sorted(r, key=lambda k: [k[0], k[1]])

        for p in r:

            # Ignore empty rows
            if len(p) == 0:
                continue

            x, y, w1, h1 = p
            if h1 > 10:

                # Extract the threshold region of interest and reshape and float array
                roi = thresh[y:y+h1,x:x+w1]
                roi_small = cv2.resize(roi,(10,10))
                roi_small = roi_small.reshape((1,100))
                roi_small = np.float32(roi_small)

                # Ask KNN model for best guess
                retval, results, neigh_resp, dists = model.findNearest(roi_small, k = 3)

                # Only take one value per digit
                if abs(last_x - x) > w1/3:
                    sail_num.append(str(int((results[0][0]))))
                last_x = x

        sail_nums.append( ''.join(sail_num))

        if len(r) <= 1:
            continue



    # Identify closest match to sail number
    sail_nums = sorted(sail_nums, key=lambda k : -len(k))
    final_sail_number = []
    for num in sail_nums:
        res = difflib.get_close_matches(num, nums, cutoff=0.66)
        if len(res) == 0 :
            continue
        final_sail_number.append(res[0])

    # Sorts all found sail numbers longest to shortest and takes the longest
    """
        TODO: This can be improved, not accurate for short sail numbers
    """
    if len(final_sail_number) != 0:
        final_sail_number = sorted(final_sail_number, key= lambda k : -len(k))
        return final_sail_number[0]
    else:
        return []


