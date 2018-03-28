import cv2
import numpy as np
import difflib
import pandas as pd

def detect_sail_number(im):
    samples = np.loadtxt('redesign_samples1.data', np.float32)
    responses = np.loadtxt('redesign_responses1.data', np.float32)
    responses = responses.reshape((responses.size, 1))
    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)


    # im = cv2.imread('../res/boats/{}.png'.format(i))
    # im = cv2.imread('../res/training images/digits.png')
    h, w = im.shape[:2]
    # if h*w < 90000:
    #     im = cv2.resize(im, (600, 600))
    out = np.zeros(im.shape,np.uint8)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)


    #_, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    mser = cv2.MSER_create()
    # mser.setMinArea(10)
    # mser.setMaxArea(200)

    # Do MSER detection, get the coordinates and bounding boxes of possible text areas
    coordinates, bboxes = mser.detectRegions(gray)
    bboxes = sorted(bboxes, key=lambda k: [k[1], k[0]])
    img = im.copy()
    last_x, last_y, last_h = w, h, 0
    results_location = []
    for cnt in bboxes:
        x,y,w1,h1 = cnt
        if w1*h1>50:
            if float(h1)/float(h) > (0.1) or float(w1)/float(w) > (0.1):
                continue
            if last_x == x and last_y ==y:
                continue
            # if abs(last_x - x) < w1/10 :#or h1 < last_h:
            #     continue
            if y > (4.0/6.0)*h:
                continue
            if h1 > 10:
                roi = thresh[y:y+h1,x:x+w1]
                roismall = cv2.resize(roi,(10,10))
                roismall = roismall.reshape((1,100))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
                string = str(int((results[0][0])))
                last_x, last_y, last_h = x, y, h1
                results_location.append([(x,y,w1,h1), string])
                cv2.rectangle(img, (x, y), (x + w1, y + h1), (0, 0, 255), 1)
    cv2.imshow('img', img)

    last_x, last_y, last_h = w, h, 0
    row_height = 0
    rows = []
    row = []
    for p in results_location:
        (x,y,w1,h1), val = p

        if h1 < 0.9*row_height and abs(y-last_y) < 0.9*row_height:
            continue
        if abs(y-last_y) > 0 and abs(x-last_x) > 2*w1:
            #print 'new row'
            rows.append(row)
            row = []
            row_height = h1
        cv2.rectangle(im, (x, y), (x + w1, y + h1), (0, 0, 255), 1)
        row.append((x,y,w1,h1))
        last_x, last_y, last_h = x, y, h1

    sail_nums = []
    for r in rows:
        result = []
        vals = []
        sail_num = []
        last_x = w
        r = sorted(r, key=lambda k: [k[0], k[1]])
        for p in r:
            if len(p) == 0:
                continue
            x, y, w1, h1 = p
            if h1 > 10:
                if abs(last_x - x) > w1/3:
                    vals.append(1)
                else:
                    vals.append(0)
                roi = thresh[y:y+h1,x:x+w1]
                roismall = cv2.resize(roi,(10,10))
                roismall = roismall.reshape((1,100))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = model.findNearest(roismall, k = 3)
                string = str(int((results[0][0])))
                result.append(string)
                cv2.rectangle(im, (x, y), (x + w1, y + h1), (0, 255, 0), 1)
                cv2.putText(out, string, (x, y + h1), 0, 1, (0, 255, 0))
                last_x = x

        for i in range(0, len(vals)):
            if vals[i] != 0:
                sail_num.append(result[i])
        sail_nums.append( ''.join(sail_num))
        if len(r) <= 1:
            continue

        d = sum([r[i][2] for i in range(0, len(r)-1)])/len(r)
        b = sum([r[i][3] for i in range(0, len(r)-1)])/len(r)
        print d, b
    numbers = pd.read_csv('../res/sample sail numbers.csv', dtype={'ID': str})
    nums = [str(num[0]) for num in numbers.values]

    # Identify closest match to sail number
    sail_nums = sorted(sail_nums, key=lambda k : -len(k))
    final_sail_number = []
    for num in sail_nums:
        res = difflib.get_close_matches(num, nums, cutoff=0.66)
        if len(res) == 0 :
            continue
        final_sail_number.append(res)

    cv2.imshow('im',im)
    cv2.imshow('out',out)
    cv2.waitKey(0)
    if len(final_sail_number) != 0:
        return final_sail_number[0][0]
    else:
        return []


    #print 'next'

for i in range(11,12):
    img = cv2.imread('../res/boats/{}.png'.format(i))
    h, w = img.shape[:2]
    #img = cv2.resize(img, (3 * w , 3*h))
    print detect_sail_number(img.copy())
    for j in range(1,8):
        h, w = img.shape[:2]
        img = cv2.resize(img, (3*w / (4), 3*h / (4 )))
        print detect_sail_number(img.copy())
    print 'end'