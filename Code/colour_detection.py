import cv2

# Finds the largest object of in given colour range within the frame
def track_buoy_by_colour(frame, lower_bound, upper_bound):
    x,y,w,h, = 0,0,0,0

    # Create a mask based on lower and upper boundaries and find contours of frame
    mask = cv2.inRange(frame, lower_bound, upper_bound)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:

        # get bounding rectangle of max contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

    return x, y, x+w, y+h


