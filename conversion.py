import numpy as np
import argparse
import cv2
import os
import imutils 

def find_contours():
  thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
  # find contours in the thresholded image and initialize the
  # shape detector
  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)

def convert(image, name):
  # First write out image
  name_ext = name + ".png"
  cv2.imwrite(name, image)
  command = "cat " + name_ext + " | pngtopnm | potrace > " + name + ".eps" 
  os.system(command)

def canny_edge(image):
  canny = cv2.Canny(image, 100, 200) 
  #minval, maxval, aperture_size (size of sobel kernel) defaults as 3, L2gradient specifies euqation for finding gradient magnitude
  return canny

def gaussian_blur(image, ksize, sigma):
  gaussian = cv2.GaussianBlur(image, (ksize, ksize), sigma)
  return gaussian


def median_blur(image, ksize):
  #ksize is aperture linear size - odd number > 1
  median = cv2.medianBlur(image, ksize)
  return median

def preprocess(image, blurring):
  # Convert image to grayscale
  grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # Blur image
  if blurring == gaussian:
    blurred = gaussian_blur(grayscale, 5, 0)
  else:
    blurred = median_blur(grayscale, ksize)
  # Apply edge detection
  edges = canny_edge(blurred)
  return edges


if __name__ == "__main__":
  try:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help = "path to the image file")
    args = vars(ap.parse_args())
    # Load image
    raster = cv2.imread(args["image"])
    processed = preprocess(raster)
    convert(processed)
  except Exception as e:
    print(e)