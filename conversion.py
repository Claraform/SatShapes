import numpy as np
import argparse
import cv2
import os
import imutils
import shapes
from math import radians, cos, sin, asin, sqrt

x_left = 1
x_right = 1
y_left = 1
y_right = 1
x_ratio = 1
y_ratio = 1
h = 1
w = 1


def haversine(lon1, lat1, lon2, lat2):
    # Convert to radians
    #print(lon1, lat1, lon2, lat2)
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    lon_range = lon2 - lon1
    lat_range = lat2 - lat1
    a = sin(lat_range/2)**2 + cos(lat1) * cos(lat2) * sin(lon_range/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of the earth (metres)
    r = 6371000
    d = c*r
    return d


def coordinates(image):
    global x_ratio, y_ratio, x_left, x_right, y_left, y_right, h, w
    h, w, _ = image.shape
    #print("h,w =", h,w)
    x_range = abs(x_left - x_right)
    y_range = abs(y_left - y_right)
    x_ratio = x_range/w
    y_ratio = y_range/h
    #print("xl,xr=", x_left,x_right)
    #print("xrange=", x_range)
    #print("xr,yr=", x_ratio, y_ratio)


def convert(image):
    # First write out image
    cv2.imwrite("vector_input.png", image)
    command = "cat vector_input.png | pngtopnm | potrace > vector_output.eps"
    os.system(command)


def canny_edge(image):
    # Calculate median of image
    m = np.median(image)
    sigma = 0.33
    # Calculate optimal canny thresholds using median of image
    lower = int(max(0, (1.0 - sigma) * m))
    upper = int(min(255, (1.0 + sigma) * m))
    # Apply canny edge detection
    canny = cv2.Canny(image, lower, upper, None, 3, True)
    #canny = cv2.Canny(image, 150, 200, None, 3, True)
    # minval, maxval, aperture_size (size of sobel kernel) defaults as 3, L2gradient specifies euqation for finding gradient magnitude
    return canny


def gaussian_blur(image, ksize, sigma):
    gaussian = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    return gaussian


def median_blur(image, ksize):
    # ksize is aperture linear size - odd number > 1
    median = cv2.medianBlur(image, ksize)
    return median


def bilateral_blur(image):
    bilateral = cv2.bilateralFilter(image, 5, 75, 75)
    return bilateral


def unsharp_mask(image):
    kernel_size = (5, 5)
    sigma = 1.0
    amount = 1.5  # amount of sharpening
    threshold = 0  # threshold for low-contrast mask
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def sobel(image):
    #cv2.FILTER_SCHARR
    sobelx = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=1)  # x
    sobely = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=1)  # y
    # Convert back to 8U
    abs_grad_x = cv2.convertScaleAbs(sobelx)
    abs_grad_y = cv2.convertScaleAbs(sobely)
    # Approximate gradient
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad


def preprocess(image, blurring, sharpen, edge_detect):
   # Convert image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur image
    if blurring == "gaussian":
        blurred = gaussian_blur(grayscale, 5, 0)
    elif blurring == "bilateral":
        blurred = bilateral_blur(grayscale)
    else:
        blurred = median_blur(grayscale, 5)
        #blurred = grayscale
    #cv2.imwrite("blurred.png", blurred)
    # Alter contrast
    alpha = 1
    beta = 1
    blurred = cv2.convertScaleAbs(blurred, alpha, beta)
    #cv2.imwrite("contrast.png", blurred)
    # Sharpen
    if sharpen == "unsharp":
        sharp = unsharp_mask(blurred)
    elif sharpen == "general":
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharp = cv2.filter2D(blurred, -1, kernel)
    else:
        sharp = blurred
    # Apply edge detection
    if edge_detect == "sobel":
        edges = sobel(sharp)
    elif edge_detect == "laplace":
        #edges = sharp
        edges = cv2.Laplacian(sharp, cv2.CV_16S, None, 3)
        edges = cv2.convertScaleAbs(edges)
    else:
        edges = canny_edge(sharp)
    cv2.imwrite("edges.png", edges)
    return edges

def circles(raster, processed, method):
# Find circles
    if method == "contour":
        output = shapes.contour_circles(raster)
        cv2.imwrite("output.png", output)
    elif method == "cht":
        #circles, output = shapes.circles(raster, processed, 20, 67, 10)
        circles, output = shapes.contour_circles(raster, processed)
        # Post-process
        convert(output) 
        # Create output file
        o_file = open("output.txt", "w")
        o_file.write("#Format: x,y,r \n")
        global x_ratio, y_ratio, x_left, y_right
        i = 1
        if circles is not None:
          for (x, y, r) in circles:
              #print("x,y,r=", x,y,r)
              print(r)
              centre_x = x_left + x*x_ratio #long1
              centre_y = y_right + y*y_ratio #lat1
              rad_point = x_left + (x+r)*x_ratio #long2
              radius = haversine(centre_x, centre_y, rad_point, centre_y)
              o_file.write("Circle " + str(i) + ": " + str(centre_x) +
                              ", " + str(centre_y) + ", " + str(radius) + "\n")
              i += 1
        o_file.close()
    else:
        global h, w
        max_d = min(h, w)
        # Create blank image with same dimensions
        output = raster.copy()
        image = raster.copy()
        output[:] = (255, 255, 255)
        for d in range(20, max_d, 2):
            locations = shapes.circle_template_match(image, processed, d)
            # Outline rectangles
            for point in zip(*locations[::-1]):
                cv2.circle(raster, (point[0]+int(d/2), point[1]+int(d/2)), int(d/2), (0, 0, 255), 4)
                cv2.circle(output, (point[0]+int(d/2), point[1]+int(d/2)), int(d/2), (0, 0, 255), 4)
        cv2.imwrite("tem.png", raster)
        cv2.imwrite("outline.png", output)


def squares(raster, processed, method):
  if method == "contour":
    output = shapes.squares(raster)
    cv2.imwrite("output.png", output)
  else:
    global h, w
    length = 51
    width = 101
    # Create blank image with same dimensions
    output = raster.copy()
    image = raster.copy()
    output[:] = (255, 255, 255)
    for y in range(50, length, 1):
      for x in range(100, width, 1):
        locations = shapes.template_match(image, processed, y, x)
        # Outline rectangles
        for point in zip(*locations[::-1]):
            cv2.rectangle(raster, point, (point[0] + x, point[1] + y), (0, 0, 255), 4)
            cv2.rectangle(output, point, (point[0] + x, point[1] + y), (0, 0, 255), 4)
    cv2.imwrite("tem.png", raster)
    cv2.imwrite("outline.png", output)

def main(raster, c_file):
    line = c_file.readline()
    line = line.split(",")
    global x_left, x_right, y_left, y_right
    y_left = float(line[0])
    x_left = float(line[1])
    y_right = float(line[2])
    x_right = float(line[3])
    c_file.close()
    coordinates(raster)
    # Set pre-processing functions
    blurring = "median"
    edge_detect = "sobel"
    sharpen = "na"
    shape = "sq_template"
    # Pre-process
    processed = preprocess(raster, blurring, sharpen, edge_detect)
    #processed = cv2.cvtColor(raster, cv2.COLOR_BGR2GRAY)
    if shape == "cht":
      circles(raster, processed, "cht")
    elif shape == "circle_template":
      circles(raster, processed, "template")
    elif shape == "circle_contour":
      circles(raster, processed, "contour")
    elif shape == "squares":
      squares(raster, processed, "contour")
    else:
      squares(raster, processed, "template")
    

if __name__ == "__main__":
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", required=True,
                        help="path to the image file")
        #args = vars(ap.parse_args())
        ap.add_argument("-c", "--coords", required=True,
                        help="path to the coords file")
        args = vars(ap.parse_args())
        # Load image
        raster = cv2.imread(args["image"])
        # Load coordinates
        c_file = open(args["coords"], "r")
        main(raster, c_file)
    except Exception as e:
        print(e)
