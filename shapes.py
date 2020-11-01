import numpy as np
import cv2
import imutils
import argparse


def create_rect_template(length, width):
    template = np.zeros((length, width, 3), np.uint8)
    cv2.rectangle(template, (width-1, 0), (0, length-1), (255, 255, 255), 1)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("template.png", template)
    return template

def create_circle_template(radius):
    template = np.zeros((radius, radius, 3), np.uint8)
    cv2.circle(template, (int(radius/2), int(radius/2)), radius, (255, 255, 255), 1)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("template.png", template)
    return template

def template_match(image, filtered, length, width):
    # Create blank image with same dimensions as input
    output = image.copy()
    output[:] = (255, 255, 255)
    # Threshold for positive matches
    threshold = 0.6
    h, w = filtered.shape
    if (w > width) and (h > length):
       # Create rectangular template
        template = create_rect_template(length, width)
        result = cv2.matchTemplate(filtered, template, cv2.TM_CCOEFF_NORMED)
        # Find all locations where match values are greater than threshold
        locations = np.where((result >= threshold))
        # Outline locations
        for point in zip(*locations[::-1]):
            cv2.rectangle(
                image, point, (point[0] + width, point[1] + length), (0, 0, 255), 4)
            cv2.rectangle(
                output, point, (point[0] + width, point[1] + length), (0, 0, 255), 4)
    # Rotate image and template match
    if (length != width) and (w > length) and (h > width):
        template = create_rect_template(width, length)
        result = cv2.matchTemplate(filtered, template, cv2.TM_CCOEFF_NORMED)
        # Find all locations where match values are greater than threshold
        locations = np.where((result >= threshold))
        # Outline locations
        for point in zip(*locations[::-1]):
            cv2.rectangle(
                image, point, (point[0] + length, point[1] + width), (0, 0, 255), 4)
            cv2.rectangle(
                output, point, (point[0] + width, point[1] + length), (0, 0, 255), 4)
    return image, output


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))


def squares(image, gray):
    squares = []
    bin = cv2.dilate(gray, None)
    bin = cv2.dilate(bin, None)
    bin = cv2.dilate(bin, None)
    cv2.imwrite("dilate.png", bin)
    contours = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.04*perimeter, True)
        if len(cnt) == 4 and cv2.contourArea(cnt) > 400: 
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos(cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4]) for i in range(4)])
            if max_cos < 0.1:
                squares.append(cnt)
    cv2.drawContours(image, squares, -1, (0, 255, 0), 3)
    return image


def circles(image, filtered, minRadius, maxRadius, minDist):
    # Create blank image with same dimensions as input
    output = image.copy()
    output[:] = (255, 255, 255)
    # Find circles
    dp = 1.5
    circles = cv2.HoughCircles(
        filtered, cv2.HOUGH_GRADIENT, dp, minDist, None, 200, 100, minRadius, maxRadius)
    if circles is not None:
        # Convert radius and centre point to int
        circles = np.round(circles[0, :]).astype("int")
        # Loop through circle coordinates
        for (x, y, r) in circles:
            # Draw detected circles onto original image (in red)
            cv2.circle(image, (x, y), r, (0, 0, 255), 4)
            # Draw detected circles onto blank image
            cv2.circle(output, (x, y), r, (0, 0, 0), 4)
    name1 = "outlined2(" + str(minDist) + ")(" + str(dp) + ").png"
    name2 = "simple2(" + str(minDist) + ")(" + str(dp) + ").png"
    cv2.imwrite(name1, image)
    cv2.imwrite(name2, output)
    return circles, output


def triangles(image, gray):
    # Find triangles
    # Create blank image with same dimensions as input
    output = image.copy()
    output[:] = (255, 255, 255)
    # Find contours in image
    # image, mode, method
    contours = cv2.findContours(
        gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    for cont in contours:
        # Compute perimeter of contour
        perimeter = cv2.arcLength(cont, True)
        # Construct contour approximation
        approx = cv2.approxPolyDP(cont, 0.04*perimeter, True)
        # Determine whether contour is a triangle
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
