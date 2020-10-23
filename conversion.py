import numpy as np
import argparse
import cv2
import os
import imutils
import shapes


def invert(image, edges):
    inverse = (1.0 / 255) * (255 - edges)
    channels = cv2.split(image)
    for channel in channels:
        channel[:] = channel * inverse
    cv2.merge(channels, edges)
    return edges


def convert(image, name):
    # First write out image
    name_ext = name + ".png"
    cv2.imwrite(name, image)
    command = "cat " + name_ext + " | pngtopnm | potrace > " + name + ".eps"
    os.system(command)


def canny_edge(image):
    # Calculate median of image
    m = np.median(image)
    sigma = 0.33
    # Calculate optimal canny thresholds using median of image
    lower = int(max(0, (1.0 - sigma) * m))
    upper = int(min(255, (1.0 + sigma) * m))
    # Apply canny edge detection
    canny = cv2.Canny(image, lower, upper, None, 3, False)
    canny2 = cv2.Canny(image, lower, upper, None, 3, True)
    cv2.imwrite("laplace.png", canny2)
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


def preprocess(image, blurring, edge_detect):
   # Convert image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur image
    if blurring == "gaussian":
        blurred = gaussian_blur(grayscale, 5, 0)
    elif blurring == "bilateral":
        blurred = bilateral_blur(grayscale)
    else:
        blurred = median_blur(grayscale, 5)
    cv2.imwrite("blurred.png", blurred)
    # Increase contrast
    alpha = 1
    beta = 1
    blurred = cv2.convertScaleAbs(blurred, alpha, beta)
    cv2.imwrite("contrast.png", blurred)
    # Sharpen
    # sharp = cv2.Laplacian(blurred, cv2.CV_8U, 1)
    #kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #sharp = cv2.filter2D(blurred, -1, kernel)
    #sharp = unsharp_mask(blurred)
    #cv2.imwrite("sharp.png", sharp)
    #edges2 = cv2.Laplacian(blurred, cv2.CV_64F)
    #cv2.imwrite("laplace.png", edges2)
    # Apply edge detection
    edges = canny_edge(blurred)
    cv2.imwrite("canny.png", edges)
    # Invert image
    #inverted = invert(grayscale, edges)
    #cv2.imwrite("inverted.png", inverted)
    # return inverted
    return edges


if __name__ == "__main__":
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", required=True,
                        help="path to the image file")
        args = vars(ap.parse_args())
        # Load image
        raster = cv2.imread(args["image"])
        # Set pre-processing functions
        blurring = "bilateral"
        edge_detect = "canny"
        # Pre-process
        processed = preprocess(raster, blurring, edge_detect)
        # Find circles
        #shapes.circles(raster, processed)
        matched = raster.copy()
        h, w = processed.shape
        for y in range(60, h-60, 1):
            for x in range(60, int(w/2), 1):
                matched = shapes.template_match(matched, processed, y, x)
        # convert(processed)
        cv2.imwrite("matched.png", matched)
    except Exception as e:
        print(e)
