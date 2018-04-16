import time
from Buoy_Detection import *
from boat_detector import *
from line_crossing import *
from boat_coords import *
from digit_recognition_new import *
import datetime

# Show user first frame and have them select the buoy then press enter
def setup(filename):

    # Set call back for buoy click
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', buoy_points)

    # Read and display first frame
    cam = cv2.VideoCapture(('../res/' + filename + '.mov'))
    ver, first = cam.read()
    cv2.imshow('image', first)
    cv2.waitKey(0)

    # Get the height and width of the frame
    h, w = first.shape[:2]

    return cam, h, w

def record_race(distance, colour, mode = 1):

    # Choose camera
    filename = 'march_8'

    cam, h, w = setup(filename)

    #out = cv2.VideoWriter('../res/sample_output.avi', -1, 23.0, (1280,720))

    #Buoy and its location are initially unknown
    buoy = []


    # Record previous line crossings so each boat is only recorded once
    line_crossing = []

    # Counters for saving the images of line crossings
    boat_crossing_counter = 0
    false_start_counter = 0

    #First frame has already been read in setup
    frame_counter = 1

    # Get the time left until race starts so false starts can be detected
    time_to_start = 0
    #time_to_start = input('How long until the race begins in minutes?')

    #Start time of video reading
    t0 = time.time()

    #Read camera input until finished
    while True:

        # Read frame, break from loop if no frames remain
        ret, frame = cam.read()
        if not ret:
            break

        # Realistically shouldn't have boats crossing within 50 pixels of each other within ~5 seconds
        if frame_counter % 120 == 0:
            line_crossing = []

        """**********Buoy*********"""
        """ 
            Only read the buoy every n frames to save processing time. In that time 
            the buoy should not move to significantly. If the buoy is not detected 
            or it has moved by more than twice its size keep it in the last known location. 
            The user can manually reset the location if the buoy is lost. 
        """
        if (frame_counter % 3 == 0 or frame_counter == 1):

            buoy_x1, buoy_y1, buoy_x2, buoy_y2, buoy = track_buoy(frame, distance, colour, frame_counter, buoy)

        if mode == 1:
            m = slope((w / 2, h), (buoy_x1, buoy_y2), 2)
            draw_buoy = [buoy_x1, buoy_y1, buoy_x2, buoy_y2]
        else:
            m = slope((w / 2, h), (buoy_x2, buoy_y2), 2)
            draw_buoy = [buoy_x2, buoy_y1, buoy_x1, buoy_y2]

        # If the buoy is at 0 ignore and read next frame
        if buoy_x1 == 0 and buoy_y1 == 0:
            # This section is purely for display to show the finish line, buoy, time (secs) and FPS
            draw_line_and_buoy(frame, draw_buoy)
            cv2.putText(frame, "TIME : " + str(round(time.time() - t0, 2)), (100, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
            cv2.putText(frame, "Total Frames : " + str(frame_counter), (100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

            cv2.imshow('image', frame)
            # out.write(frame)
            cv2.waitKey(1)
            frame_counter += 1
            continue

        """**********Boats*********"""
        """ 
            Only want to detect boats every n frames and on first frame
            We need to create a seperate tracker for each boat
        """
        if frame_counter % 3 == 0 or frame_counter == 1 :

            # Handle left to right and right to left direction of travel
            if mode == 1:
                boats, coords = detect_boats(frame[0:h, 0:int(buoy_x1)+100])
            else:
                boats, coords = detect_boats(frame[0 : h,  0 : w])

            trackers = []
            for obj in range(0, len(boats), 1):

                # Create a tracker for each boat in the image that is found
                trackers.append( cv2.TrackerMedianFlow_create())

        # Track boats that were detected on the last detection
        for i, c in enumerate(coords):

                new_points, boat_img = get_extreme_points(frame, trackers, mode, i, c)

                if new_points == -1:
                    continue

                for p in new_points:

                    # Ignore points near points that have been detected as crossing the line since last cleared line_crossing
                    if check_previous_points(line_crossing, p) == -1:
                        continue

                    # Show the point on the screen - For testing purposes
                    cv2.circle(frame, p, 1, (0,0,255), thickness=1)

                    # Calculate the slope of the point to the bottom left of the buoy
                    m1 = slope(p, (buoy_x1, buoy_y2), 3)

                    # If the slope of boat point is in an acceptable range it has crossed the line
                    if m1 > m-.3 and m1 < m+.3:

                        # Add point of intersection to detected points to avoid repetition
                        line_crossing.append(p)

                        # Create a copy of the image to save as proof of intersection and draw buoy,
                        # line and line crossing point
                        proof_img = frame.copy()
                        cv2.circle(proof_img, p, 2, (255, 0, 0), 2)
                        draw_line_and_buoy(proof_img, draw_buoy)

                        # If the boat image is big enough attempt to read sail numbers
                        if (boat_img.shape[1] > 50):
                            sail_number = detect_sail_number(boat_img.copy())
                            if len(sail_number) == 0:
                                print 'Unable to recognise sail number'
                                """
                                    TO DO: Raise a flag for human intervention regarding unidentified boat
                                """

                            # If the intersection occurs before the race starts it is a false start
                            if not has_race_started(t0, time_to_start):
                                print 'Boat {} false started'.format(sail_number)
                                false_start_counter += 1
                                cv2.imwrite('../res/Screen-Shots/False Starts/{}.png'.format(false_start_counter), proof_img)
                                continue

                            # Write finish time and sail number to output file as results
                            file = open('../res/finishes/' + filename + '.txt', "a")
                            file.write('Boat  with sail number {} finished at {} with a time of {}\n'.format(sail_number, datetime.datetime.now().time(), float(frame_counter)/24))
                            file.close()

                            boat_crossing_counter += 1
                            cv2.imwrite('../res/Screen-Shots/Finishes/{}.png'.format(boat_crossing_counter), proof_img)
                            print 'Intersection at: {} by boat : {}'.format(p, sail_number)

                            # No need to continue to loop if this boat has already crossed the line
                            break

        # This section is purely for display to show the finish line, buoy, time (secs) and FPS
        draw_line_and_buoy(frame, draw_buoy)
        cv2.putText(frame, "TIME : " + str(round(time.time() - t0, 2)), (100, 110),
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

# time_to_start is given in minutes
def has_race_started(t0, time_to_start):
    if math.ceil((time.time() - t0)) > time_to_start * 60:
        return True
    return False

def draw_line_and_buoy(img,  buoy):
    cv2.rectangle(img, (int(buoy[0]), int(buoy[1])), (int(buoy[2]), int(buoy[3])), (0, 255, 0), 1)
    cv2.line(img, (img.shape[1] / 2, img.shape[0]), (int(buoy[0]), int(buoy[3])), (0, 0, 255), 1)

def get_extreme_points(frame, trackers, mode, i, c):
    # Initialize tracker with first frame and bounding box
    t = trackers[i]
    # t.init(frame, (c[0], c[1]+(c[3]-c[1])/2, c[2], c[3]))

    t.init(frame, (c[0], c[1], c[2], c[3]))

    # Update tracker
    ok, bbox = t.update(frame)

    # if tracking succeeded update boat points
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[2]), int(bbox[3]))
    else:
        return -1, None

    # Ensure that none of the points are invalid
    if p1[0] < 0 or p1[1] < 0 or p2[0] < 0 or p2[1] < 0:
        return -1, None

    # Extract boat image from original image for processing
    boat_img = frame[p1[1]:p2[1], p1[0]:p2[0]]

    # Discard boats that are too small to process meaningfully
    if len(boat_img) < 50:
        return -1, None

    # Get the extreme points of the boat for line crossing
    extreme_points = get_extreme_point(boat_img.copy(), mode)

    # If no points are found continue
    if extreme_points == None:
        return -1, None

    # Extreme points are relative to boat_img not the entire frame, add boat coordinates to extreme points
    return  [(extreme_points[i][0] + p1[0], extreme_points[i][1] + p1[1]) for i in range(0, len(extreme_points))], boat_img

def check_previous_points(line_crossing, p):

    if len(line_crossing) != 0:

        # Ignore points where the sum of the differences of the points is less than 50
        for l in line_crossing:
            if abs((l[1] + l[0]) - (p[1] + p[0])) < 100:
                return -1