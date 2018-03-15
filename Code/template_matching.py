import cv2

def match_template(img, template, last_location, distance, manual_change):

    # Height and width of buoy
    h, w = template.shape[:2]

    # If buoy was not found last frame and it is a small buoy and the user hasn't clicked the screen,
    # use a smaller search window
    if sum(last_location) != 0 and distance > 80 and not manual_change:

        # Ensures that any non-zero values are rounded to zero
        f = lambda a: (abs(a) + a) / 2

        # Create a 300 x 300 pixel area around the last buoy location
        y1, y2, x1, x2 = int(last_location[1] - (150)), int(last_location[3] + (150)),\
                         int(last_location[0] - (150)), int(last_location[2] + (150))

        # Round non-zero locations to zero
        y1, y2, x1, x2 = f(y1), f(y2), f(x1), f(x2)

        # Crop original image
        img1 = img[y1: y2, x1: x2]

        # If the new image has invalid dimensions search entire image
        if img1.shape[0] < w or img1.shape[1] < h:

            res = cv2.matchTemplate(img.copy(), template, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = (min_loc[0], min_loc[1])
            bottom_right = (top_left[0] + w, top_left[1] + h)

        # If the image is bigger than the last buoy try find the buoy inside of it
        else:

            res = cv2.matchTemplate(img1.copy(), template, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = (min_loc[0] + int(last_location[0] - (2*w)), min_loc[1] + int(last_location[1] - (2*h)))
            bottom_right = (top_left[0] + w, top_left[1] + h)

    # Otherwise search entire image for the buoy
    else:

        res = cv2.matchTemplate(img.copy(), template, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = (min_loc[0], min_loc[1])
        bottom_right = (top_left[0] + w, top_left[1] + h)

    return top_left[0], top_left[1], bottom_right[0], bottom_right[1]
