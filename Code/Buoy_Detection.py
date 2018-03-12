import cv2
import imutils
import numpy as np
from template_matching import *
from colour_detection import track_buoy_by_colour

yCoord = 100
xCoord = 100
count = 0
distance = 0
colour = ''
upper_bound = np.array([255, 255, 255])
lower_bound = np.array([0, 0, 0])

# mouse callback function to take user input
# as the position of the buoy
def buoy_points(event,x,y,flags,param):
    global xCoord, yCoord, count
    if event == cv2.EVENT_LBUTTONDOWN:
        xCoord= x
        yCoord =  y
        count = 0

def track_buoy(frame, buoy = []):
    global count, colour, distance
    manual_change = False
    if xCoord != 100 and yCoord != 100:
        #Find the buoy - only performed once at start up
        if count < 1:
            distance = input('Approximately how far away is the buoy?')
            colour = raw_input('What is the main colour of the buoy?')
            #Get bounds for mask
            get_colour(colour)
            #get approx size of buoy
            size = calc_range(distance)
            #Get buoy image
            buoy = frame[yCoord - size:yCoord + size, xCoord - size:xCoord + size].copy()
            count+=1
            manual_change = True
        """ 
            Use correlation to detect buoy in frame, then use colour detection to find the largest dominant colour
            in the image (should be the buoy), if the center of each detected buoy is within a reasonable distance 
            then it is most likely the buoy we want
        """
        x1, y1, x2, y2 = match_template(frame, buoy, [lower_bound, upper_bound])
        if distance < 100:
            (lower, upper) = get_colour_for_tracking(colour)
            x1_c, y1_c, x2_c, y2_c = track_buoy_by_colour(frame, lower, upper)
            center_1 = (int((x2-x1)/2), int((y2-y1)/2))
            center_2 = (int((x2_c - x1_c) / 2), int((y2_c - y1_c) / 2))
            d = (abs(center_1[0]- center_2[0]), abs(center_1[0]- center_2[0]))
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            # cv2.rectangle(frame, (x1_c, y1_c), (x2_c, y2_c), (0,0,255), 2)
            # cv2.imshow('buoy', frame)
            if d[0] > x2-x1 or d[1] > y2 -y1:
                return 0, 0, 0, 0, buoy, manual_change
        return float(x1),float(y1),float(x2),float(y2), buoy, manual_change
    else:
        return 0,0,0,0, buoy, manual_change


def calc_range(distance):
    size = 3000
    return int(size/(2*distance))

def get_colour(colour):
    global lower_bound, upper_bound
    if colour.lower() == 'red':
        lower_bound = np.array([150, 150, 100])
        upper_bound = np.array([255, 255, 255])
    # if colour.lower() == 'white':
    #     lower_bound = np.array([100, 100, 200])
    #     upper_bound = np.array([255, 255, 255])
    if colour == 'white':
        lower_bound = np.array([0, 0, 50])
        upper_bound = np.array([50, 50, 255])
    if colour.lower() == 'yellow':
        lower_bound = np.array([225, 180, 0])
        upper_bound = np.array([255, 255, 170])
    if colour.lower() == 'orange':
        lower_bound = np.array([204, 85, 0])
        upper_bound = np.array([255, 245, 238])
    if colour.lower() == 'black':
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([75, 75, 75])

def buoy_within_boat(boat_coords, buoy_coords):
    x_boat, y_boat, x1_boat, y1_boat = boat_coords
    x_buoy, y_buoy, x1_buoy, y1_buoy = buoy_coords

    if (x_buoy > x_boat and x_buoy < x1_boat) and (y_buoy > y_boat and y_buoy < y1_boat):
        return True
    return False

def get_colour_for_tracking(colour):
    return {
        'red': (np.array([0, 0, 50]),  np.array([50, 50, 255])),
    # elif colour == 'white':
    #     lower_bound = np.array([0, 0, 0])
    #     upper_bound = np.array([0, 0, 255])
        'white' : (np.array([100, 100, 200]), np.array([255, 255, 255]))
    }[colour]