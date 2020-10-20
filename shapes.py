import numpy as np
import cv2
import imutils
import argparse


def squares():
    # Find squares
    x = 0


def circles(image, gray):
    # Create blank image with same dimensions as input
    output = image.copy()
    output[:] = (255, 255, 255)
    # Find circles
    dp = 1.5
    minDist = 20
    minRadius = 0
    maxRadius = 0
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, minDist)
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
    name1 = "outlined2(" + str(minDist) + ")(" + str(dp) + ").png"
    name2 = "simple2(" + str(minDist) + ")(" + str(dp) + ").png"
    cv2.imwrite(name1, image)
    cv2.imwrite(name2, output)


def triangles(image, gray):
    # Find triangles
    # Create blank image with same dimensions as input
    output = image.copy()
    output[:] = (255, 255, 255)
    # Find contours in image
    #image, mode, method
    contours = cv2.findContours(
        gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    for cont in contours:
      #Compute perimeter of contour
      perimeter = cv2.arcLength(cont, True)
      #Construct contour approximation
      approx = cv2.approxPolyDP(cont, 0.04*perimeter, True)
      #Determine whether contour is a triangle
      if len(approx) == 3:
        cv2.drawContours(image, [cont], -1, (0, 0, 255), 4)
        cv2.drawContours(output, [cont], -1, (0, 0, 255), 4)

    cv2.imwrite("triangle.png", image)
    cv2.imwrite("skeleton_tri.png", output)

def rectangles():
    # Find rectangles
    x = 0


if __name__ == "__main__":
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", required=True,
                        help="path to the image file")
        args = vars(ap.parse_args())
        # Load image
        image = cv2.imread(args["image"])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        triangles(image, gray)
    except Exception as e:
        print(e)
