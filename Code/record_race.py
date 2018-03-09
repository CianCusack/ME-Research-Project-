from Buoy_Detection import *
import math
import time

from imutils import contours

from Buoy_Detection import *
from boat_detector import *
from line_crossing import *
from colour_detection import *
from boat_coords import *
from digit_recognition import *


def setup(cam):
    ## Show user first frame and have them select the buoy
    display = cv2.namedWindow('image')
    cv2.setMouseCallback('image', buoy_points)
    ver, first = cam.read()
    cv2.imshow('image', first)
    cv2.waitKey(3000)
    cv2.destroyWindow('image')

def record_race():
    #Choose camera
    #cam = cv2.VideoCapture(0)
    #cam = cv2.VideoCapture('../res/sailing.mov')
    #cam = cv2.VideoCapture('../res/olympic_sailing_short.mp4')
    cam = cv2.VideoCapture('../res/new_race_2.mov')
    #cam = cv2.VideoCapture('../res/KishRace6.mp4')

    setup(cam)

    #out = cv2.VideoWriter('../res/sample_output.avi', -1, 23.0, (1280,720))
    #Buoy is initially unknown
    buoy = []
    counter = 0
    #First frame has already been read in setup
    frame_counter = 1
    last_x1, last_y1, last_x2, last_y2 = 0,0,0,0
    buoy_x1, buoy_y1, buoy_x2, buoy_y2 = 0,0,0,0
    time_to_start = 1
    #time_to_start = input('How long until the race begins in minutes?')
    #Start time of video reading


    t0 = time.time()
    #Read camera input until finished
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        # if frame_counter < 100 and frame_counter > 10:
        #     frame_counter+=1
        #     continue
        #Check if race has begun
        h, w = frame.shape[:2]


        """**********Buoy*********"""
        if (frame_counter % 10 == 0 or frame_counter == 1):
            buoy_x1, buoy_y1, buoy_x2, buoy_y2, buoy = track_buoy(frame.copy(), buoy)
            buoy_size = buoy.shape[:2]
            if (buoy_x1 == 0.0 and buoy_y1 == 0.0) or ((abs(buoy_x1 - last_x1) > buoy_size[0] or abs(buoy_y1 - last_y1) > buoy_size[1]) and frame_counter > 1):
                # lower, upper = get_colour('red')
                # buoy_x1, buoy_y1, buoy_x2, buoy_y2 = track_buoy_by_colour(frame, lower, upper)
                # if ((buoy_x1 - last_x1 > buoy_size[0] or buoy_y1 - last_y1 > buoy_size[1])):
                buoy_x1, buoy_y1, buoy_x2, buoy_y2 = last_x1, last_y1, last_x2, last_y2
            if frame_counter > 1:
                tracker = cv2.TrackerKCF_create()

                tracker.init(frame, (buoy_x1, buoy_y1, buoy_x2, buoy_y2))
        if frame_counter > 10 and frame_counter % 3 == 0:

            ok, bbox = tracker.update(frame)
            #if tracking succeeded update boat points
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[2]), int(bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                if(abs(p1[0] - last_x1) < buoy_size[0] or abs(p1[1] - last_y1) < buoy_size[1]):
                    buoy_x1, buoy_y1 = p1
                    buoy_x2, buoy_y2 = p2
        # Assuming that the center of the camera/video is one end of start/finish line
        m = slope((w/2, h), (buoy_x1, buoy_y2))
            # If buoy position jumps more than halfway across image ignore
        """
            ***** Revisit this solution especially concerning max buoy size
        """
        max_buoy_width = max_buoy_height = 60
        # if buoy_x1 != 0 and buoy_y1 != 0:
        #     if(abs(buoy_x1 - last_x1) <= max_buoy_width and abs(buoy_y1 - last_y1) <= max_buoy_height) or last_y1 == 0:
        # cv2.rectangle(frame, (int(buoy_x1), int(buoy_y1)), (int(buoy_x2), int(buoy_y2)), (0,255,0), 1)
        # cv2.line(frame, (w / 2, h), (int(buoy_x1), int(buoy_y2)), (0, 0, 255), 1)
        last_x1, last_y1, last_x2, last_y2 = buoy_x1, buoy_y1, buoy_x2, buoy_y2
        # else:
        #     cv2.putText(frame, "No buoy", (200, 300), cv2.FONT_HERSHEY_COMPLEX_SMALL, 10, (0, 0, 0))
        #     counter += 1
        #     if counter > 10:
        #         last_y1 = 0

        """**********Boats*********"""
        """ Only want to detect boats every n frames and on first frame
            We need to create a seperate tracker for each boat
            """
        if frame_counter % 23 == 0 or frame_counter == 1:
            boats, coords = detect_boats(frame[0:h, 0:int(buoy_x1+20)])
            trackers = []
            for obj in range(0, len(boats), 1):
                # Initialize tracker with first frame and bounding box
                tracker = cv2.TrackerMedianFlow_create()
                trackers.append(tracker)

        # Track boats that were detected on the last detection
        elif frame_counter % 3 == 0:
            for i, c in enumerate(coords):
                # Initialize tracker with first frame and bounding box
                if c[2] > buoy_x1:
                    continue
                t = trackers[i]
                t.init(frame, (c[0], c[1], c[2], c[3]))

                # Update tracker
                ok, bbox = t.update(frame)
                #if tracking succeeded update boat points
                if ok:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[2]), int(bbox[3]))
                    #cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                else:
                    # Tracking failure
                    cv2.putText(frame, "Tracking failure detected", (100, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                # Ensure that none of the points are invalid
                if p1[0] < 0 or p1[1] < 0 or p2[0] < 0 or p2[1] < 0:
                    continue
                img = frame[p1[1]:p2[1], p1[0]:p2[0]].copy()


                # Discard boats that are too small to process meaningfully
                if len(img) < 50:
                    continue
                # Get the extreme points of the boat for line crossing
                extreme_points = get_extreme_point(img)
                if extreme_points == None:
                    continue
                new_points = [(extreme_points[i][0] + p1[0], extreme_points[i][1]+p1[1]) for i in range(0, len(extreme_points))]
                for p in new_points:
                    if p[0] > buoy_x1:
                        new_points.remove(p)
                # Check that the boat has crossed the line by checking slope of new_points
                for p in new_points:
                    cv2.circle(frame, p, 5, (0,0,255), thickness=3)
                    m1 = slope(p, (buoy_x1, buoy_y2))
                    if m1 == m:
                        if not has_race_started(t0, time_to_start):
                            print 'Boat false started'
                            continue
                        if(img.shape[1] > 50):
                            sail_number = get_sail_number(img)
                        cv2.putText(frame, "Intersection", (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,100), 2)
                        proof_img = frame.copy()
                        cv2.circle(proof_img, p,2, (255,0,0), 2)
                        cv2.line(proof_img, (w / 2, h), (int(buoy_x1), int(buoy_y2)), (0, 0, 255), 1)
                        file = open('../res/finishes.txt', "a")
                        cv2.imwrite('../res/Screen-Shots/line_crossing.png', proof_img)
                        print 'Intersection'
                        file.write('Boat  with sail number {} finished at {} \n'.format(sail_number, time.time() - t0))
                        file.close()
        # for i, c in enumerate(coords):
        #     test = frame[c[1]:c[3], c[0]:c[2]]
        #     cv2.imwrite('../res/test/test{}.png'.format(i+1), test)
        #     ##Draw rough triangle around boats
        #     # bottom_left = (c[0], c[3])
        #     # bottom_right = (c[2], c[3])
        #     # top_middle = (c[0]+(c[2] - c[0])/2, c[1])
        #     # cv2.circle(frame, (bottom_left), 2, (255, 0, 0), 2)
        #     # cv2.circle(frame, (top_middle), 2, (255, 0, 0), 2)
        #     # cv2.circle(frame, (bottom_right), 2, (255, 0, 0), 2)
        #     # cv2.line(frame, bottom_left, top_middle, (0, 0, 255))
        #     # cv2.line(frame, top_middle, bottom_right, (0, 0, 255))
        #     # cv2.line(frame, bottom_left, bottom_right, (0, 0, 255))
        #
        #     """
        #         Creates array of points along the rightmost edge of the boat bounding box
        #         if the point has the same slope to the buoy as the finish line then it is
        #         at the finish line and will records as point of intersection
        #     """
        #     y_points = np.arange(c[1], c[3], 0.01)
        #     for p in y_points:
        #         #Ignore points beyond the buoy for the moment
        #         if p > buoy_y1:
        #             m1 = slope((c[2], p), (buoy_x1, buoy_y2))
        #             if m1 == m:
        #                 cv2.putText(frame, "Intersection", (100, 100),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,100), 2)
        #                 cv2.circle(frame, (c[2],int(p)),2, (255,0,0), 2)
        #                 cv2.imwrite('../res/Screen-Shots/line_crossing.png', frame)
        #                 file.write('Boat {} finished at{} \n'.format(i, time.time() - t0))

            ## Detect sail numbers
            # for b in boats:
            #     if not b:
            #         boat_copy = locate_numbers(b)
            #         cv2.imshow("boat", boat_copy)
        """ This section is purely for display to show the time, total frames and type of tracker"""
        cv2.rectangle(frame, (int(buoy_x1), int(buoy_y1)), (int(buoy_x2), int(buoy_y2)), (0, 255, 0), 1)
        cv2.line(frame, (w / 2, h), (int(buoy_x1), int(buoy_y2)), (0, 0, 255), 1)
        cv2.putText(frame, "TIME : " + str(round(time.time() - t0, 2)), (100, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        cv2.putText(frame, 'Median Flow' + " Tracker", (100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        cv2.putText(frame, "Total Frames : " + str(frame_counter), (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
        cv2.imshow('frame', frame)
        #out.write(frame)
        cv2.waitKey(1)
        frame_counter += 1
    print 'Total frames: {}'.format(frame_counter)
    print 'Average frame rate: {} FPS'.format(frame_counter/(time.time() - t0))
    cam.release()
    #out.release()

def locate_numbers(boat):

    lower_white = np.array([0, 0, 255 - 100])
    upper_white = np.array([255, 100, 255])
    (h, w) = boat.shape[:2]
    ## Sail numbers normally in the middle third of the boat
    boat = boat[h / 3:(2 * h) / 3, 0:w]

    hsv = cv2.cvtColor(boat, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(boat, boat, mask=mask)
    boat = cv2.cvtColor(boat, cv2.COLOR_BGR2GRAY)
    _, boat = cv2.threshold(boat, 75, 110, 0)
    _, cnts, _ = cv2.findContours(boat.copy(), cv2.RETR_LIST,
                                  cv2.CHAIN_APPROX_SIMPLE)

    #Sort contours from left to right
    cnts = contours.sort_contours(cnts, method="left-to-right")[0]
    rois = []
    for c in cnts:
        if cv2.contourArea(c) > 100 or cv2.contourArea(c) < 50:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # region of interest is only within the contours
        roi = boat[y:y + h, x:x + w].copy()
        roi = cv2.resize(roi, (200, 75))
        ## Convert from gray to hsv and back to gray to get the hsv for masking
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ## Sail numbers usually black or red - black in this case
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([0, 100, 100])
        mask = cv2.inRange(hsv, lower_black, upper_black)
        roi = cv2.bitwise_and(roi, roi, mask=mask)
        ##Store current roi in array of regions of interest
        rois.append(roi)
        ## Drawing here merely for clarity can be removed in future
        cv2.rectangle(boat, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(boat, c, -1, (0, 255, 0), 1)

    ## Shows the regions of interest for half a second each
    i = 0
    for roi in rois:
        cv2.imshow("roi {}".format(i), roi)
        i += 1
    cv2.waitKey(500)
    while i >= 0:
        cv2.destroyWindow("roi {}".format(i))
        i -= 1
    return boat

def has_race_started(t0, time_to_start):
    if math.ceil((time.time() - t0)) == time_to_start * 60:
        return True
    return False
def get_colour(colour):
    if colour == 'red':
        lower_bound = np.array([0, 0, 50])
        upper_bound = np.array([50, 50, 255])
    elif colour == 'white':
        lower_bound = np.array([0, 0, 50])
        upper_bound = np.array([50, 50, 255])
    else:
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([255, 255, 255])

    return lower_bound, upper_bound