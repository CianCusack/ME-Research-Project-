import datetime
import cv2

e1 = cv2.getTickCount()

def draw_circle(event,x,y,flags,param):
    global xCoord, yCoord
    if event == cv2.EVENT_LBUTTONDOWN:
        print x,y
        xCoord= x
        yCoord =  y

def slope(x, y):
    xPoint = 640
    yPoint = 360
    m = 0
    if x-xPoint != 0:
        m = float((y-yPoint)/(x-xPoint))
    return m





yCoord = 1000
xCoord = 1000
time = 0
counter = 1

camera = cv2.VideoCapture('res/KishRace6.mp4')

display = cv2.namedWindow('Boat Feed')
cv2.setMouseCallback('Boat Feed', draw_circle)
# initialize the first frame in the video stream
firstFrame = None
# loop over the frames of the video
while True:
    (grabbed, frame) = camera.read()
    if counter %6 ==0:

        # grab the current frame and initialize the occupied/unoccupied
        # text


        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if not grabbed:
            break


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # if the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = gray
            continue
        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 700:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)


            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


            e2 = cv2.getTickCount()
            time = (e2 - e1) / cv2.getTickFrequency()


            #if time > 21:
            if slope(xCoord, yCoord) == slope(float(x+w), float(y+h)):
                print "Intersection", x+w,y+h, time, counter
                #exit(0)


        cv2.line(frame, (1280/ 2, 720), (xCoord, yCoord), (0, 0, 255), 1)
        cv2.imshow("Boat Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the loop
        if key == ord("q"):
            break
    counter = counter + 1

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()



