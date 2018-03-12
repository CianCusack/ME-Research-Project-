import numpy as np
import cv2

net = cv2.dnn.readNetFromCaffe("../deep learning/MobileNetSSD_deploy.prototxt.txt",
                                   "../deep learning/MobileNetSSD_deploy.caffemodel")

def detect_boats(image):
    global net
    confidence_thresh = 0.1
    # Really only care about boats but can detect others
    classes = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "dining table",
		"dog", "horse", "motorbike", "person", "potted plant", "sheep",
		"sofa", "train", "tv monitor"]

    # load our serialised model from disk
    # net = cv2.dnn.readNetFromCaffe("../deep learning/MobileNetSSD_deploy.prototxt.txt",
    #                                "../deep learning/MobileNetSSD_deploy.caffemodel")

    # current frame is used to construct an input blob by resizing to a
    # fixed 300x300 pixels and then normalizing it
    # changing resizing alters accuracy and as a result performance
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    boats = []
    boat_positions = []
    for i in np.arange(0, detections.shape[2]):
        #Get the confidence of the object found
        confidence = detections[0, 0, i, 2]

        # Ignore detections below confidence or errors where background is detected
        # i.e. confidence should always be between threshold and 1, ignore all else
        if confidence > confidence_thresh and confidence <= 1:
            # Compute the position of the boat and add it to the boat_positions array
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            boat_positions.append([startX, startY, endX, endY])
            boat = image[startY:endY, startX: endX].copy()
            boats.append(boat)

    return boats, boat_positions
