# import the necessary packages
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
def check_glass(image):
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # threshold the image to reveal light regions in the
    # blurred image
    thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)[1]

    # perform a series of erosions and dilations to remove
    # any small blobs of noise from the thresholded image
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)


    labels = measure.label(thresh, connectivity=2, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue

        # otherwise, construct the label mask and count the
        # number of pixels
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 300:
            mask = cv2.add(mask, labelMask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts)==0:
        return "ok"
    else:
        c = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        h1, w1, _ = image.shape
        xp1 = 0
        yp1 = 150
        xp2 = int(w1 / 2.55)
        yp2 = h1
        a = 0
        for (i, c) in enumerate(cnts):
            a = a+1
        if a >=1:
            if FindPoint(xp1, yp1, xp2,
                         yp2, x, y,x+w,y+h):
                print("aaaaaaaaa")
                return "face_fake"
            else:
                print("bbbbbbb")

                return "glass"
        else:
            return "ok"
        # draw the bright spot on the image
        # (x, y, w, h) = cv2.boundingRect(c)
        # ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        #
        # cv2.circle(image, (int(cX), int(cY)), int(radius),
        #            (0, 0, 255), 3)
        # cv2.putText(image, "#{}".format(i + 1), (x, y - 15),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        #
def FindPoint(x1, y1, x2,
              y2, xp1, yp1,xp2,yp2 ):
    if (xp1 >= x1 and xp1 <= x2 and
            yp1 >= y1 and yp1 <= y2) and (xp2 >= x1 and xp2 <= x2 and
            yp2 >= y1 and yp2 <= y2):
        return True
    else:
        return False
