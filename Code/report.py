import cv2

cam = cv2.VideoCapture('../res/KishRace6BoatCloseShort.mp4')

while(1):
    ret, img = cam.read()
    if not ret:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,127,255,0)[1]
#    thresh = cv2.threshold(gray, 194, 15, cv2.THRESH_BINARY_INV)[1]
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        cv2.drawContours(img, c, -1, (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.imwrite('../res/contour_example.png', img)
    cv2.waitKey(1000/23)
cam.release()