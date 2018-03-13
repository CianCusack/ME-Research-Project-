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
    #cv2.destroyWindow('image')

def record_race():
    #Choose camera
    #cam = cv2.VideoCapture(0)
    #cam = cv2.VideoCapture('../res/sailing.mov')
    #cam = cv2.VideoCapture('../res/olympic_sailing_short.mp4')
    #cam = cv2.VideoCapture('../res/new_race_1.mov')
    cam = cv2.VideoCapture('../res/KishRace6BoatCloseShort.mp4')

    setup(cam)

    #out = cv2.VideoWriter('../res/sample_output.avi', -1, 23.0, (1280,720))
    #Buoy is initially unknown
    buoy = []
    boat_crossing_counter = 0
    false_start_counter = 0
    #First frame has already been read in setup
    frame_counter = 1
    last_x1, last_y1, last_x2, last_y2 = 0,0,0,0
    buoy_x1, buoy_y1, buoy_x2, buoy_y2 = 0,0,0,0
    time_to_start = 0
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
        if (frame_counter % 5 == 0 or frame_counter == 1):
            buoy_x1, buoy_y1, buoy_x2, buoy_y2, buoy, user_change = track_buoy(frame.copy(), buoy)
            buoy_size = buoy.shape[:2]
            if ((buoy_x1 == 0.0 and buoy_y1 == 0.0) or ((abs(buoy_x1 - last_x1) > 2*buoy_size[0] or abs(buoy_y1 - last_y1) > 2*buoy_size[1]) and frame_counter > 1)) and not user_change:
                buoy_x1, buoy_y1, buoy_x2, buoy_y2 = last_x1, last_y1, last_x2, last_y2

        # Assuming that the center of the camera/video is one end of start/finish line
        m = slope((w/2, h), (buoy_x1, buoy_y2))


        last_x1, last_y1, last_x2, last_y2 = buoy_x1, buoy_y1, buoy_x2, buoy_y2


        """**********Boats*********"""
        """ Only want to detect boats every n frames and on first frame
            We need to create a seperate tracker for each boat
            """
        if frame_counter % 23 == 0 or frame_counter == 1:
            boats, coords = detect_boats(frame[0:h, 0:int(buoy_x1)])
            trackers = []
            line_crossing = set([])
            for obj in range(0, len(boats), 1):
                # Initialize tracker with first frame and bounding box
                tracker = cv2.TrackerMedianFlow_create()
                trackers.append(tracker)

        # Track boats that were detected on the last detection
        elif frame_counter % 1 == 0:
            for i, c in enumerate(coords):
                # Initialize tracker with first frame and bounding box
                if c[2] > buoy_x1 or c[2] < w/2:
                    continue
                t = trackers[i]
                t.init(frame, (c[0], c[1]+(c[3]-c[1])/2, c[2], c[3]))

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

                # If the rightmost point is before the start of the line continue
                if p2[0] < w/ 2:
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
                    if p[0] > buoy_x1 or p[0] < w / 2:
                        continue
                    if len(line_crossing) != 0:
                        for l in line_crossing:
                            if abs(l[0] - p[0]) < 50 or abs(l[1] - p[1]) < 50:
                                p = None
                                break
                    if p == None:
                        continue
                    cv2.circle(frame, p, 5, (0,0,255), thickness=3)
                    m1 = slope(p, (buoy_x1, buoy_y2))
                    if m1 == m:
                        line_crossing.add(p)
                        proof_img = frame.copy()
                        cv2.circle(proof_img, p, 2, (255, 0, 0), 2)
                        cv2.rectangle(proof_img, (int(buoy_x1), int(buoy_y1)), (int(buoy_x2), int(buoy_y2)),
                                      (0, 255, 0), 1)
                        cv2.line(proof_img, (w / 2, h), (int(buoy_x1), int(buoy_y2)), (0, 0, 255), 1)
                        if not has_race_started(t0, time_to_start):
                            print 'Boat false started'
                            false_start_counter += 1
                            cv2.imwrite('../res/Screen-Shots/False Starts/{}.png'.format(false_start_counter), proof_img)
                            continue
                        if(img.shape[1] > 50):
                            sail_number = get_sail_number(img)
                        cv2.putText(frame, "Intersection", (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,100), 2)
                        file = open('../res/finishes.txt', "a")
                        file.write('Boat  with sail number {} finished at {} \n'.format(sail_number, time.time() - t0))
                        file.close()
                        boat_crossing_counter += 1
                        cv2.imwrite('../res/Screen-Shots/Finishes/{}.png'.format(boat_crossing_counter), proof_img)
                        print 'Intersection at: {}'.format(p)

                        break

        """ This section is purely for display to show the time, total frames and type of tracker"""
        cv2.rectangle(frame, (int(buoy_x1), int(buoy_y1)), (int(buoy_x2), int(buoy_y2)), (0, 255, 0), 1)
        cv2.line(frame, (w / 2, h), (int(buoy_x1), int(buoy_y2)), (0, 0, 255), 1)
        cv2.putText(frame, "TIME : " + str(round(time.time() - t0, 2)), (100, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        cv2.putText(frame, 'Median Flow' + " Tracker", (100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        cv2.putText(frame, "Total Frames : " + str(frame_counter), (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
        cv2.imshow('image', frame)
        #out.write(frame)
        cv2.waitKey(1)
        frame_counter += 1
    print 'Total frames: {}'.format(frame_counter)
    print 'Average frame rate: {} FPS'.format(frame_counter/(time.time() - t0))
    cam.release()
    #out.release()


def has_race_started(t0, time_to_start):
    if math.ceil((time.time() - t0)) > time_to_start * 60:
        return True
    return False
