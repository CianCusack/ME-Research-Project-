import cv2
#file_name = "res/Screen-Shots/Intersection1.png"

cap = cv2.VideoCapture('res/KishRace6BoatCloseShort.mp4')
#cap = cv2.VideoCapture(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

while(1):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)

    fgmask = fgbg.apply(gray)
    fgmaskClean = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)


    #print thresh
    (_,cnts, _) = cv2.findContours(fgmaskClean.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 500:
            continue

        #detect_shapes(c)
        extRight = tuple(c[c[:, :, 0].argmax()][0])

        #c = c.astype("float")
        #c = c.astype("int")
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)



    cv2.imshow('mask',fgmask)
    cv2.imshow('clean frame', fgmaskClean)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()