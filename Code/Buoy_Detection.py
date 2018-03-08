import cv2
import imutils
import numpy as np
from template_matching import *

yCoord = 100
xCoord = 100
count = 0
upper_bound = np.array([255, 255, 255])
lower_bound = np.array([0, 0, 0])

# mouse callback function to take user input
# as the position of the buoy
def buoy_points(event,x,y,flags,param):
    global xCoord, yCoord
    if event == cv2.EVENT_LBUTTONDOWN:
        xCoord= x
        yCoord =  y

def track_buoy(frame, buoy = []):
    global count
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
        x1, y1, x2, y2 = match_template(frame, buoy, [lower_bound, upper_bound])
        """ 
            TO DO: Need to check that returned buoy image is the same
            color as the buoy check mask is in same area same size
        """
        return float(x1),float(y1),float(x2),float(y2), buoy
    else:
        return 0,0,0,0, buoy


def calc_range(distance):
    size = 3000
    return int(size/(2*distance))

def get_colour(colour):
    global lower_bound, upper_bound
    if colour.lower() == 'red':
        lower_bound = np.array([150, 150, 100])
        upper_bound = np.array([255, 255, 255])
    if colour.lower() == 'white':
        lower_bound = np.array([100, 100, 200])
        upper_bound = np.array([255, 255, 255])
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