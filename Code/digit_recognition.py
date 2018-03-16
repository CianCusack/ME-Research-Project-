from four_point_transform import *
from recognise_numbers import *
from imutils import rotate
import difflib
import pandas as pd

"""
    TO DO: Fix inefficiency of this awful code
"""

# Attempt to identify row of sail numbers
def get_sail_number_line(boxes, img):

    xs = []
    ys = []
    hs = []
    for b in boxes:
        xs.append(b[0])
        ys.append(b[1])
        hs.append(b[3])

    top_line = [0, 0, 0, 0]
    bottom_line = [0, 0, 0, 0]
    for b in boxes:
        if b[0] == min(xs):
            top_line[0] = (b[0], int(b[1] * 0.975))
            top_line[1] = (b[0], int((b[1] + b[3]) * 1.025))
        if b[1] == min(ys):
            top_line[2] = (b[0] + b[2], int(b[1] * 0.95))
            top_line[3] = (b[0] + b[2], int((b[1] + b[3]) * 1))
        if b[0] == max(xs):
            bottom_line[0] = (b[0] + b[2], int(b[1] * 0.975))
            bottom_line[1] = (b[0] + b[2], int((b[1] + b[3]) * 1.025))
        if b[1] == max(ys):
            bottom_line[2] = (b[0], b[1])
            bottom_line[3] = (b[0], b[1] + b[3])

    sail_imgs = []
    mask = np.zeros(img.shape, dtype=np.uint8)
    roi_corners = np.array([[(top_line[0]), (top_line[2]), (top_line[3]), (top_line[1])]], dtype=np.int32)
    ignore_mask_color = (255,) * img.shape[2]
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)

    img_mask = img[top_line[2][1]:top_line[1][1], top_line[0][0]:top_line[3][0]]
    mask = mask[top_line[2][1]:top_line[1][1], top_line[0][0]:top_line[3][0]]
    masked_image = cv2.bitwise_and(img_mask, mask)
    cv2.imwrite('../res/Sail Numbers/top.png', masked_image)
    sail_imgs.append(masked_image)

    mask = np.zeros(img.shape, dtype=np.uint8)
    roi_corners = np.array([[(bottom_line[0]), (bottom_line[2]), (bottom_line[3]), (bottom_line[1])]], dtype=np.int32)
    ignore_mask_color = (255,) * img.shape[2]
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)

    img_mask = img[bottom_line[0][1]:bottom_line[3][1], bottom_line[3][0]:bottom_line[0][0]]
    mask = mask[bottom_line[0][1]:bottom_line[3][1], bottom_line[3][0]:bottom_line[0][0]]
    masked_image = cv2.bitwise_and(img_mask, mask)
    cv2.imwrite('../res/Sail Numbers/bottom.png', masked_image)
    sail_imgs.append(masked_image)

    return sail_imgs


def detect_digits(img):

    # Ignore images that are too small
    height, width = img.shape[:2]
    if height * width < 200:
        return None, -1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get MSER, and set parameters - may need to play with max
    # and min value possibly with regard to the size of the boat
    mser = cv2.MSER_create()
    mser.setMinArea(60)
    mser.setMaxArea(200)

    # Do MSER detection, get the coordinates and bounding boxes of possible text areas
    coordinates, bboxes = mser.detectRegions(gray)

    # If no text found, return
    if len(coordinates) == 0:
        return None, -1

    # Sort the coordinates
    coordinates, _ = sort_contours(coordinates, 'top-to-bottom')

    coords = []
    boxes = []
    angles = []
    last_max_x = width
    last_height = 0

    digit_counter = 0
    row_counter = 0

    row= []

    digits = []
    for coord in coordinates:

        #Get bounding box of the text area and remove areas too small to process
        bbox = cv2.boundingRect(coord)
        x,y,w1,h1 = bbox
        if w1< 8 or h1 < 15 or w1 > h1 or h1/w1 > 10:
            continue

        coords.append(coord)
        boxes.append(bbox)

        # Get rotated bounding box coordinates
        rect = cv2.minAreaRect(coord)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # for p in box:
        #     cv2.circle(img, (p[0], p[1]), 2, (255,255,0), 2)

        cv2.rectangle(img, (x, y), (x+w1, y+h1), (255,255,0), 1)

        # Get the min and max y coordinates
        min_x = width
        max_x = 0
        min_y = height
        max_y = 0

        for p in box:
            if min_x > p[0]:
                min_x = p[0]
            if max_x < p[0]:
                max_x = p[0]
            if min_y > p[1]:
                min_y = p[1]
            if max_y < p[1]:
                max_y = p[1]

        # Text areas in covering same digit
        if abs(last_max_x - max_x) < 5:
            continue

        # Ignore text areas that may be inside other text areas
        if last_height - (max_y - min_y) > (1.0/4.0)*last_height:
            continue

        # If the x coordinate increases it is the start of a new row
        if last_max_x < max_x:
            print '********** New row detected **********'
            print 'Detecting digits on digits set'
            for r in row:
                digits.append(guess_numbers(cv2.resize(r, (2*r.shape[1], 2*r.shape[0]))))
            print ''.join(digits)
            # Read in sail numbers of boats in the race and convert to string array
            numbers = pd.read_csv('../res/sample sail numbers.csv', dtype={'ID': str})
            nums = [str(num[0]) for num in numbers.values]

            # Identify closest match to sail number
            print (difflib.get_close_matches(''.join(digits), nums))

            print 'Detecting digits on mirrored digits'
            digits = []
            for r in row:
                r = cv2.flip(r, 1)
                digits.append(guess_numbers(cv2.resize(r, (2*r.shape[1], 2*r.shape[0]))))

            print ''.join(digits)
            # Read in sail numbers of boats in the race and convert to string array
            numbers = pd.read_csv('../res/sample sail numbers.csv', dtype={'ID': str})
            nums = [str(num[0]) for num in numbers.values]

            # Identify closest match to sail number
            print (difflib.get_close_matches(''.join(digits), nums))
            print 'Detecting digits on mirrored digits'
            digits = []

            digits = []
            row = []

        # Extract the detected digit
        digit = img[min_y : max_y, min_x : max_x]


        # Rotate the extracted digit
        rotated = rotate(digit, rect[2])

        digit_counter += 1

        row.append(digit)
        last_max_x = max_x
        last_height = digit.shape[0]

        theta =  abs(rect[2]/45)
        if theta == 0 or theta == 1:
            continue
        angles.append(theta)

    cv2.imshow('img', img)
    cv2.waitKey(0)

    if len(angles) == 0:
        return None, -1

    average_angle =  sum(angles)/len(angles)
    mode = 0

    # Determine if the angle of the sail numbers is small enough to ignore
    if 2 - average_angle < 0.8 or 2-average_angle > 1.2:
        sail_imgs = get_sail_number_line(boxes, img)
        mode = 1

    # Extract each of the images of a number
    else:
        xs = []
        ys = []
        hs = []
        ws = []
        for b in boxes:
            xs.append(b[0])
            ys.append(b[1])
            ws.append(b[2])
            hs.append(b[3])

        top_xs= []
        top_ys = []
        for b in boxes:
            if b[1] > min(ys) + max(hs):
                    continue
            top_xs.append(b[0])
            top_ys.append(b[1])
        sail_imgs = [img[min(top_ys): max(top_ys)+max(hs), min(top_xs):max(top_xs)+max(ws)].copy()]

    return sail_imgs, mode



def get_sail_number(img):

    # Attempt to get the sail numbers
    imgs, mode = detect_digits(img)

    #Rotate if necessary
    if mode == 1:
        imgs = four_point_transform(imgs, 0)

    # If no numbers found, return
    if mode == -1 or imgs == None:
        return []

    # Attempt to recognise the numbers
    digits = []
    for i, img in enumerate(imgs):
        if i == 0 and mode == 1:

            # One of the identified sail number areas will be mirrored
            img = cv2.flip(img, 1)
        digits.append(recognise_digits(img))

    # If no sail numbers are recognised, return
    if digits[0] == '' and len(digits) == 1:
        return []

    # Read in sail numbers of boats in the race and convert to string array
    numbers = pd.read_csv('../res/sample sail numbers.csv', dtype={'ID': str})
    nums = [str(num[0]) for num in numbers.values]

    # Identify closest match to sail number
    results = []
    results.append(difflib.get_close_matches(digits[0], nums))

    return results[0]


# print get_sail_number(cv2.imread('../res/sail_numbers_cropped.jpg'))
print get_sail_number(cv2.imread('../res/Screen-Shots/Finishes/2.png'))