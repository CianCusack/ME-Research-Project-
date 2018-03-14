import numpy as np
from template_matching import *
from colour_detection import track_buoy_by_colour


# Initialise global variables
yCoord = 296
xCoord = 777
count = 0
distance = 0
colour = ''

# Mouse callback function to take user input as the position of the buoy
def buoy_points(event,x,y,flags,param):
    global xCoord, yCoord, count
    if event == cv2.EVENT_LBUTTONDOWN:
        xCoord= x
        yCoord =  y
        count = 0

def track_buoy(frame, buoy = [], last_location = []):
    global count, colour, distance
    manual_change = False
    if xCoord != 100 and yCoord != 100:
        #Find the buoy - only performed when user clicks the screen to select the buoy
        if count ==0:
            distance = input('Approximately how far away is the buoy?')
            colour = raw_input('What is the main colour of the buoy?')
            # distance = 150
            # colour = ''
            #get approx size of buoy
            size = calc_range(distance)
            #Get buoy image
            buoy = frame[yCoord - size:yCoord + size, xCoord - size:xCoord + size].copy()
            count += 1
            manual_change = True
        """ 
            Use correlation to detect buoy in frame, then use colour detection to find the largest dominant colour
            in the image (should be the buoy), if the center of each detected buoy is within a reasonable distance 
            then it is most likely the buoy we want
        """
        # Use correlation to detect buoy in frame
        x1, y1, x2, y2 = match_template(frame, buoy, last_location, distance)

        # At greater distances the colour/size of the buoy can make it difficult to find using colour detection
        if distance < 100:

            # Get the colour boundaries based on user input colour, if no input just use correlation method
            (lower, upper) = get_colour_for_tracking(colour)
            if sum(lower) == 0 and sum(upper) == 0:
                return float(x1), float(y1), float(x2), float(y2), buoy, manual_change

            # Use colour detection to detect buoy in frame
            x1_c, y1_c, x2_c, y2_c = track_buoy_by_colour(frame, lower, upper)

            # Determine center of the area returned by both methods
            center_1 = (int((x2-x1)/2), int((y2-y1)/2))
            center_2 = (int((x2_c - x1_c) / 2), int((y2_c - y1_c) / 2))

            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            # cv2.rectangle(frame, (x1_c, y1_c), (x2_c, y2_c), (0,0,255), 2)
            # cv2.imshow('frame', frame)
            # cv2.waitKey(0)
            # Find the distance between the two centroids
            d = (abs(center_1[0]- center_2[0]), abs(center_1[0]- center_2[0]))

            # If the distance is too large assume neither reading is reliable and return no buoy found
            if d[0] > x2-x1 or d[1] > y2 -y1:
                return 0, 0, 0, 0, buoy, manual_change
        return float(x1),float(y1),float(x2),float(y2), buoy, manual_change
    else:
        return 0,0,0,0, buoy, manual_change


def calc_range(distance):
    size = 3000
    return int(size/(2*distance))


def get_colour_for_tracking(colour):
    return {
        'red': (np.array([0, 0, 50]),  np.array([50, 50, 255])),
        'white' : (np.array([100, 100, 200]), np.array([255, 255, 255])),
        '' : (np.array([0, 0, 0]), np.array([0, 0, 0]))
    }[colour]