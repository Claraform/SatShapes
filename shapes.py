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


def create_circle_template(diameter):
    template = np.zeros((diameter, diameter, 3), np.uint8)
    cv2.circle(template, (int(diameter/2), int(diameter/2)),
               (int(diameter/2)-1), (255, 255, 255), 1)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("template.png", template)
    return template


def template_match(image, filtered, length, width):
    # Threshold for positive matches
    threshold = 0.56
    h, w = filtered.shape
    if (w > width) and (h > length):
       # Create rectangular template
        template = create_rect_template(length, width)
        result = cv2.matchTemplate(filtered, template, cv2.TM_CCOEFF_NORMED)
        # Find all locations where match values are greater than threshold
        locations = np.where((result >= threshold))
    return locations


def circle_template_match(image, filtered, diameter):
    # Threshold for positive matches
    threshold = 0.35
    template = create_circle_template(diameter)
    result = cv2.matchTemplate(filtered, template, cv2.TM_CCOEFF_NORMED)
    # Find all locations where match values are greater than threshold
    locations = np.where((result >= threshold))
    return locations

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))


def squares(image):
    # Pad image to ignore border contour
    img2 = np.pad(image.copy(), ((4,4), (4,4), (0,0)), 'edge')
    img = cv2.GaussianBlur(img2, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(
                    gray, thrs, 255, cv2.THRESH_BINARY)
            bin = cv2.bitwise_not(bin)
            contours, _hierarchy = cv2.findContours(
                bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max(
                        [angle_cos(cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4]) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    cv2.drawContours(img2, squares, -1, (0, 0, 255), 4)
    # Remove padding
    img = img2[4:-4,4:-4,:]
    return img


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
    cv2.imwrite("outlined.png", image)
    cv2.imwrite("simple.png", output)
    return circles, output


def contour_circles(image):
    # Pad image to ignore border contour
    img2 = np.pad(image.copy(), ((4,4), (4,4), (0,0)), 'edge')
    img = cv2.GaussianBlur(img2, (5, 5), 0)
    circles = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=3)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(
                    gray, thrs, 255, cv2.THRESH_BINARY)
            bin = cv2.bitwise_not(bin)
            contours, _hierarchy = cv2.findContours(
                bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.03*cnt_len, True)
                if len(cnt) >= 8 and cv2.contourArea(cnt) > 200 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max(
                        [angle_cos(cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4]) for i in range(4)])
                    if max_cos < 0.1:
                        circles.append(cnt)
    cv2.drawContours(img2, circles, -1, (0, 0, 255), 4)
    # Remove padding
    img = img2[4:-4,4:-4,:]
    return img


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
