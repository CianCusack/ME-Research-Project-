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
    #mser.setMinArea(60)
    #mser.setMaxArea(200)

    # Do MSER detection, get the coordinates and bounding boxes of possible text areas
    coordinates, bboxes = mser.detectRegions(gray)

    # If no text found, return
    if len(coordinates) == 0:
        return None, -1

    # Sort the coordinates
    coordinates, _ = sort_contours(coordinates, 'top-to-bottom')
    # coordinates, _ = sort_contours(coordinates, 'right-to-left')


    img_temp = img.copy()

    for coord in coordinates:
        bbox = cv2.boundingRect(coord)
        x, y, w1, h1 = bbox
        if w1 < 8 or h1 < 15 or w1 > h1 or h1 / w1 > 10:
            continue


    coords = []
    boxes = []
    angles = []
    last_max_x = width
    last_height = 0

    digit_counter = 0

    row= []

    digits = []
    for coord in coordinates:

        #Get bounding box of the text area and remove areas too small to process
        bbox = cv2.boundingRect(coord)
        x,y,w1,h1 = bbox
        if w1< 8 or h1 < 15 or w1 > h1 or h1/w1 > 10:
            continue



        # Get rotated bounding box coordinates
        rect = cv2.minAreaRect(coord)
        box = cv2.boxPoints(rect)
        box = np.int0(box)


        # for p in box:
        #     cv2.circle(img, (p[0], p[1]), 2, (255,255,0), 2)



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
        if abs(last_height - (max_y - min_y)) < (8.0/9.0)*last_height:
            continue


        skip = False
        for p in boxes:
            if p == bbox:
                skip = True
                break
            if abs(p[0] - x) < w1/7:
                skip = True
                break
        if not skip:
            coords.append(coord)
            boxes.append(bbox)

        cv2.rectangle(img, (x, y), (x + w1, y + h1), (255, 255, 0), 1)
    cv2.imshow('img_temp', img)
    cv2.waitKey(0)
    #print len(boxes)
    #     # If the x coordinate increases it is the start of a new row
    #     if last_max_x < max_x:
    #         print '********** New row detected **********'
    #         print 'Detecting digits on digits set'
    #         for r in row:
    #             # cv2.imshow('row', r)
    #             # cv2.waitKey(0)
    #             digits.append(guess_numbers(cv2.resize(r, (2*r.shape[1], 2*r.shape[0]))))
    #         print ''.join(digits)
    #         # Read in sail numbers of boats in the race and convert to string array
    #         numbers = pd.read_csv('../res/sample sail numbers.csv', dtype={'ID': str})
    #         nums = [str(num[0]) for num in numbers.values]
    #
    #         # Identify closest match to sail number
    #         print (difflib.get_close_matches(''.join(digits), nums))
    #
    #         print 'Detecting digits on mirrored digits'
    #         digits = []
    #         for r in row:
    #             r = cv2.flip(r, 1)
    #             digits.append(guess_numbers(cv2.resize(r, (2*r.shape[1], 2*r.shape[0]))))
    #
    #         print ''.join(digits)
    #         # Read in sail numbers of boats in the race and convert to string array
    #         numbers = pd.read_csv('../res/sample sail numbers.csv', dtype={'ID': str})
    #         nums = [str(num[0]) for num in numbers.values]
    #
    #         # Identify closest match to sail number
    #         print (difflib.get_close_matches(''.join(digits), nums))
    #         print 'Detecting digits on mirrored digits'
    #
    #         digits = []
    #         row = []
    #
    #     # Extract the detected digit
    #     digit = img[min_y : max_y, min_x : max_x]
    #
    #
    #     # Rotate the extracted digit
    #     rotated = rotate(digit, rect[2])
    #
    #     digit_counter += 1
    #
    #     row.append(digit)
    #     last_max_x = max_x
    #     last_height = digit.shape[0]
    #
    #     theta =  abs(rect[2]/45)
    #     if theta == 0 or theta == 1:
    #         continue
    #     angles.append(theta)
    #
    #
    # if len(angles) == 0:
    #     return None, -1




    temp_points = []
    final_points = []
    sail_images = []

    """
        Sort the general text areas in to rows, then sort the rows by x position
    """

    points = sorted(boxes, key=lambda k: [k[1], k[0]])

    for p in points:
        x, y, w, h = p
        if y in temp_points:
            continue
        #print 'new row'


        for p1 in points:
            x1, y1, w1, h1 = p1
            if abs(y-y1) < h and (x1 != x and y1 != y):

                temp_points.append(y1)
                final_points.append(p1)

        final_points.append(p)
    no_dups = []
    for p in final_points:
        if p not in no_dups:
            no_dups.append(p)

    no_dups = sorted(no_dups, key=lambda k: [k[1], k[0]])
    last_x, last_y = 0,0
    row_length = 0
    rows = []
    for p in no_dups:
        x, y, w, h = p
        #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)


        if abs(last_x - x) < 10 and abs(last_y-y) < 10:
            continue

        if abs(last_y - y) > h/2:
            # print 'new row'
            rows.append(row_length)
            row_length = 0

        if row_length >= 6:
            continue

        last_x, last_y = x, y

        row_length += 1
        #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)
        sail_images.append(img[y: y+h, x:x+w])
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
    # cv2.imshow('img_temp', img)
    # cv2.waitKey(0)
    mode =0
    # average_angle =  sum(angles)/len(angles)
    # mode = 0
    #
    # # Determine if the angle of the sail numbers is small enough to ignore
    # if 2 - average_angle < 0.8 or 2-average_angle > 1.2:
    #     sail_imgs = get_sail_number_line(boxes, img)
    #     mode = 1
    #
    # # Extract each of the images of a number
    # else:
    #     xs = []
    #     ys = []
    #     hs = []
    #     ws = []
    #     for b in boxes:
    #         xs.append(b[0])
    #         ys.append(b[1])
    #         ws.append(b[2])
    #         hs.append(b[3])
    #
    #     top_xs= []
    #     top_ys = []
    #     for b in boxes:
    #         if b[1] > min(ys) + max(hs):
    #                 continue
    #         top_xs.append(b[0])
    #         top_ys.append(b[1])
    #     sail_imgs = [img[min(top_ys): max(top_ys)+max(hs), min(top_xs):max(top_xs)+max(ws)].copy()]

    return sail_images, mode, rows



def get_sail_number(img):

    # Attempt to get the sail numbers
    imgs, mode, rows = detect_digits(img)

    #Rotate if necessary
    if mode == 1:
        imgs = four_point_transform(imgs, 0)

    # If no numbers found, return
    if mode == -1 or imgs == None:
        return []

    # Attempt to recognise the numbers
    digits = []
    for i, img in enumerate(imgs[:12]):
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        digits.append(recognise_digits(img))

    # If no sail numbers are recognised, return
    if digits == '' and len(digits) == 1:
        return []
    norm = []
    rot = []
    rot_mir = []
    for d in (digits):
        if len(d) == 0:
            continue
        norm.append(d[0][0])
        rot.append(d[0][1])
        rot_mir.append(d[0][2])
    dig_arrays = [norm, rot, rot_mir]
    results = []
    for d in dig_arrays:
        prev = 0
        sail_numbers = []
        for r in rows:
            i = 0
            final = []
            while i < r:
                if i+prev < len(d):
                    final.append(d[i+prev])
                i += 1
            #print final
            prev+=r
            sail_numbers.append(''.join(final))
        print sail_numbers
        # Read in sail numbers of boats in the race and convert to string array
        numbers = pd.read_csv('../res/sample sail numbers.csv', dtype={'ID': str})
        nums = [str(num[0]) for num in numbers.values]

        # Identify closest match to sail number

        for num in sail_numbers:
            res = difflib.get_close_matches(num, nums, cutoff=0.85 )
            if len(res) == 0:
                continue
            results.append(res)

        #print d

        # if len(results) == 0:
        #     return []
    return results


# sail_nums =  get_sail_number(cv2.imread('../res/sail_numbers_cropped.jpg'))
#sail_nums =  get_sail_number(cv2.imread('../res/test/test1.png'))
#sail_nums =  get_sail_number(cv2.imread('../res/boats/6.png'))

sail_nums =  get_sail_number(cv2.imread('../res/training images/old/training_image.png'))
guess_numbers(sail_nums)
for num in sail_nums:
    print num
# img = cv2.imread('../res/boat.jpg')
# h, w = img.shape[:2]
# #img = img[h/3 : (2*h)/3 , 0: w]
#
# print get_sail_number(img)