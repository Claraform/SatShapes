import numpy as np
import cv2
import imutils 


def squares():
    # Find squares
    x = 0


def circles(image):
    # Create blank image with same dimensions as input
    output = image.copy()
    output[:] = (255, 255, 255)
    # Find circles
    minDist = 100
    minRadius = 0
    maxRadius = 0
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.5, minDist)
    if circles is not None:
        # Convert radius and centre point to int
        circles = np.round(circles[0, :]).astype("int")
        # Loop through circle coordinates
        for (x, y, r) in circles:
            # Draw detected circles onto original image (in red)
            #image, centre, radius, colour, thickness
            cv2.circle(image, (x, y), r, (0, 0, 255), 4)
            # Draw detected circles onto blank image
            cv2.circle(output, (x, y), r, (0, 0, 0), 4)
            # Fill in area?


def triangles(image):
    # Find triangles
    #Find contours in image
    #image, mode, method
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)


def rectangles():
    # Find rectangles
    x = 0
