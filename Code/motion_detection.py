import math
import cv2
import numpy as np

def calulate_points(x,y):
    xPoint = 640.0
    yPoint = 720.0
    listPoints = []
    m = float((y - yPoint) / (x - xPoint))
    #print m
    m = math.ceil(m*100)/100
    #print m
    xSteps = np.arange(xPoint, x, 0.1)
    ySteps = np.arange(y, yPoint, 0.1)
    for i in xSteps:
        for j in ySteps:
            if(i!= xPoint and j!=yPoint):
                slope = float((j - yPoint) / (i - xPoint))
                slope = math.ceil(slope*100)/100
                #print slope
                if slope == m:
                    """print "start"
                    print i
                    print j
                    i = math.ceil(x*100)/100
                    j = math.ceil(y*100)/100
                    print "after rounding"
                    print i
                    print j"""

                    listPoints.append((i,j))
    return listPoints


def draw_circle(event,x,y,flags,param):
    global xCoord, yCoord
    if event == cv2.EVENT_LBUTTONDOWN:
        print x,y
        xCoord= x
        yCoord =  y

def slope(x, y):
    xPoint = float(640)
    yPoint = float(720)
    m = float(0)
    if x-xPoint != 0:
        m = float((y-yPoint)/(x-xPoint))
        m = math.ceil(m * 100) / 100
    return m


def line_detection_contours(cnts, points, frame):
    intersect_points = []
    count = 0


    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 500:
            continue

        #detect_shapes(c)
        extRight = tuple(c[c[:, :, 0].argmax()][0])

        c = c.astype("float")
        c = c.astype("int")
        #cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

        if extRight in points:
            print "Intersection",
            count += 1
            intersect_points.append(extRight)
    return intersect_points

def line_detection_features(pts, points, frame, old_points):
    intersect_points = []
    count = 0


    for p in pts:
        for pt in points:
            threshold =  (abs(p[0] - pt[0])+abs(p[1]-pt[1]))
            if threshold < 2.5 and p not in old_points:
                """if len(intersect_points) != 0:
                    temp = intersect_points[-1]
                    if abs((p[0] + p[1]) - (temp[0] + temp[1])) > 100:
                        print "Intersection"
                        count += 1
                        intersect_points.append(p)
                else:"""
                print "First Intersection"
                count += 1
                intersect_points.append(p)
    print list(set(intersect_points))
    if len(intersect_points) != 0:
        for p in (set(intersect_points)):
            old_points.append(p)
    print old_points
    return list(set(intersect_points))
"""
def detect_shapes(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 3:
        shape = "triangle"
        print shape
"""

def motion_detection():

    old_points = []
    yCoord = 293.0
    xCoord = 775
    count = 0
    counter = 1
    points = calulate_points(xCoord, yCoord)
    camera = cv2.VideoCapture('res/KishRace6BoatCloseShort.mp4')


    # initialize the first frame in the video stream
    firstFrame = None
    # loop over the frames of the video
    while True:
        (grabbed, frame) = camera.read()
        if not grabbed:
            break
        if counter % 18 == 0:
            firstFrame = None
        if counter % 1 == 0:

            #for p in points:
             #   cv2.circle(frame, (int(p[0]), int(p[1])), 1, (0,255,0))

            cv2.line(frame, (640, 720), (int(xCoord), int(yCoord)), (255, 255, 0), 2 )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # if the first frame is None, initialize it
            if firstFrame is None:
                firstFrame = gray
                counter+=1
                continue
            # compute the absolute difference between the current frame and
            # first frame
            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

            # dilate the thresholded image to fill in holes, then find contours
            # on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=2)
            (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
            fast = cv2.FastFeatureDetector_create(10)

            # find and draw the key points
            kp = fast.detect(frame, None)
            pts = [p.pt for p in kp]

            #Using contours
            #intersect_points = line_detection_contours(cnts, points, frame)

            #Using fast feature detection

            cv2.drawKeypoints(frame, kp, frame, color=(255, 0, 0))
            intersect_points = line_detection_features(pts[:70], points, frame, old_points)

            for p in intersect_points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 5, (0, 0, 255), thickness=2)
                count +=1
                cv2.imwrite("res/Screen-Shots/Intersection" + `count` + ".png", frame)

                if count == 3:
                    camera.release()
                    break
            cv2.imshow("Boat Feed", frame)
            cv2.waitKey(1)
            counter += 1

        else:
            counter += 1
        #




def display_video():
    yCoord = 293
    xCoord = 775

    camera = cv2.VideoCapture('res/KishRace6BoatCloseShort.mp4')

    display = cv2.namedWindow('Boat Feed')
    cv2.setMouseCallback('Boat Feed', draw_circle)
    # initialize the first frame in the video stream
    firstFrame = None
    # loop over the frames of the video
    while True:
        (grabbed, frame) = camera.read()
        if not grabbed:
            break

        #cv2.line(frame, (640, 720), (xCoord, yCoord), (0, 255, 0), 2)
        cv2.line(frame, (1280 / 2, 720), (xCoord, yCoord), (0, 0, 255), 1)
        cv2.imshow("Boat Feed", frame)
        cv2.waitKey(1)

    camera.release()
    cv2.destroyAllWindows()