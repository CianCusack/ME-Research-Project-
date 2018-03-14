import cv2

# Find the extreme points on a boat
def get_extreme_point(img):

    # Ignore images that are too small to process
    h, w = img.shape[:2]

    if h < 10 or w < 10:
        return None


    # Detect contours in the image
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray.copy(), 127, 255, cv2.THRESH_BINARY_INV)
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.THRESH_BINARY_INV)

    # for c in cnts:
    #     cv2.drawContours(img, c, -1, (255,0,0))
    # cv2.imshow('boat', img)
    # cv2.waitKey(0)
    temp_x = []
    temp_y = []

    # Only care about the bottom third of the boat as that will protrude the most
    if h < 100 and w < 100:
        thresh = 0
    else:
        thresh = (2.0/3.0)*float(h)

    # Re-format list so that it is a list of points
    flat_list = [internal_item for sublist in cnts for item in sublist for internal_item in item]

    # Add points to lists that are above threshold value
    for f in flat_list:
        if f[1] > thresh:
            temp_x.append(f[0])
            temp_y.append(f[1])

    # if no contours are found break
    if len(cnts) == 0 or len(temp_x) == 0:
        return None

    # Get the points that are within 90% of the maximum x point
    points = []
    max_x = max(temp_x)
    for x,y in zip(temp_x, temp_y):
        if x < max_x*0.9:
            continue
        points.append((x, y))

    # Sort the points by x coord from high to low and return top 10
    sorted_by_x = sorted(points, key=lambda tup: tup[0])
    sorted_by_x = sorted_by_x[::-1]

    return sorted_by_x[:10]
