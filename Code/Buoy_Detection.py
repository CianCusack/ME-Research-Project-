import numpy as np
from template_matching import *
from colour_detection import track_buoy_by_colour

# Initialise global variables
yCoord = 296
xCoord = 777
last_x1, last_y1, last_x2, last_y2 = 0, 0, 0, 0
buoy_x1, buoy_y1, buoy_x2, buoy_y2 = 0, 0, 0, 0
count = 0

# Mouse callback function to take user input as the position of the buoy
def buoy_points(event,x,y,flags,param):
    global xCoord, yCoord, count
    if event == cv2.EVENT_LBUTTONDOWN:
        xCoord= x
        yCoord =  y
        count = 0

def track_buoy(frame, distance, colour, frame_counter, buoy = []):
    global count, last_x1, last_y1, last_x2, last_y2
    manual_change = False
    last_location = [last_x1, last_y1, last_x2, last_y2]



    #Find the buoy - only performed when user clicks the screen to select the buoy
    if count ==0:

        # Get buoy size
        size = calc_range(distance)

        # Get buoy image
        buoy = frame[yCoord - size:yCoord + size, xCoord - size:xCoord + size].copy()

        count += 1
        manual_change = True
    """ 
        Use correlation to detect buoy in frame, then use colour detection to find the largest dominant colour
        in the image (should be the buoy), if the center of each detected buoy is within a reasonable distance 
        then it is most likely the buoy we want
    """
    # Use correlation to detect buoy in frame
    x1, y1, x2, y2 = match_template(frame, buoy, last_location, distance, manual_change)

    buoy_size = buoy.shape[:2]
    if ((x1 == 0.0 and y1 == 0.0) or (
            abs(x1 - last_x1) > buoy_size[0] or abs(y1 - last_y1) > buoy_size[
        1]) and frame_counter > 1) and not manual_change:  # and frame_counter %23 != 0:
        x1, y1, x2, y2 = last_x1, last_y1, last_x2, last_y2

    # Assuming that the center of the camera/video is one end of start/finish line

    # Remember the last location of the buoy
    last_x1, last_y1, last_x2, last_y2 = x1, y1, x2, y2
    # At greater distances the colour/size of the buoy can make it difficult to find using colour detection
    if distance < 100:

        # Get the colour boundaries based on user input colour, if no input just use correlation method
        (lower, upper) = get_colour_for_tracking(colour)
        if sum(lower) == 0 and sum(upper) == 0:
            return float(x1), float(y1), float(x2), float(y2), buoy

        # Use colour detection to detect buoy in frame
        x1_c, y1_c, x2_c, y2_c = track_buoy_by_colour(frame, lower, upper)

        # Determine center of the area returned by both methods
        center_1 = (int(((x2-x1)/2)+ x1), int(((y2-y1)/2)+y1))
        center_2 = (int(((x2_c - x1_c)/ 2) + x1_c) , int(((y2_c - y1_c) / 2) +y1_c))

        # Find the distance between the two centroids
        d = (abs(center_1[0]- center_2[0]), abs(center_1[0]- center_2[0]))

        # If the distance is too large assume neither reading is reliable and return no buoy found
        if d[0] > x2-x1 or d[1] > y2 -y1:
            return 0, 0, 0, 0, buoy


    return float(x1),float(y1),float(x2),float(y2), buoy


def calc_range(distance):
    return int(1500.0/distance)


"""
    TO DO: Add more colours
"""
def get_colour_for_tracking(colour):
    return {
        'red': (np.array([0, 0, 50]),  np.array([50, 50, 255])),
        'white' : (np.array([100, 100, 200]), np.array([255, 255, 255])),
        '' : (np.array([0, 0, 0]), np.array([0, 0, 0]))
    }[colour]