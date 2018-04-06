from imutils import rotate
import numpy as np
import cv2
import glob

# Read in training images
imgs = []
for img in glob.glob("../res/training images/redesign1/*.jpg"):
    imgs.append(cv2.imread(img))

# Create empty place holder for data and labels
samples = np.empty((0, 100))
responses = np.empty((0, 100))

for img in imgs:

    # Iterate between +-20 degrees rotation on each image
    for i in range(20, -20, -1):

        im = rotate(img, i)
        h1, w1 = im.shape[:2]

        # Removes black areas from the rotated image
        im = im[h1/5 : (4*h1)/5, w1/5 : (4*w1)/5]

        # Obtain contours
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Only accept numbers as valid input
        keys = [i for i in range(48, 58)]

        for cnt in contours:

            # Only consider viable contours
            if cv2.contourArea(cnt) > 40:

                # Get the bounding box
                [x, y, w, h] = cv2.boundingRect(cnt)


                if h  > 28:

                    # Highlight the digit
                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)

                    # Extract and save the digit
                    roi = thresh[y:y + h, x:x + w]
                    roismall = cv2.resize(roi, (10, 10))
                    cv2.imshow('norm', im)

                    # Wait for user input
                    key = cv2.waitKey(0)

                    if key == 27:  # (escape to quit)
                        break
                    elif key in keys:

                        # Take user input and save label and image data
                        responses = np.append(responses, int(chr(key)))
                        print responses[-1]
                        sample = roismall.reshape((1, 100))
                        samples = np.append(samples, sample, 0)


# Reshape and float labels
responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))

# Export data set and labels
np.savetxt('redesign_samples1.data',samples)
np.savetxt('redesign_responses1.data',responses)

print "training complete"

