'''
Simple "Square Detector" program.
Loads several images sequentially and tries to find squares in each image.
Note: this code is from https://github.com/opencv/opencv/blob/master/samples/python/squares.py and was not written by me
Currently being used for testing purposes only
'''

# Python 2/3 compatibility
from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))


def find_squares(img):
    img = cv.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv.Canny(gray, 0, 50, apertureSize=5)
                bin = cv.dilate(bin, None)
            else:
                _retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
            contours, _hierarchy = cv.findContours(
                bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv.arcLength(cnt, True)
                cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max(
                        [angle_cos(cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4]) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to the image file")
    args = vars(ap.parse_args())
    # Load image
    img = cv.imread(args["image"])
    squares = find_squares(img)
    cv.drawContours(img, squares, -1, (0, 255, 0), 3)
    cv.imwrite('squares.png', img)
    ch = cv.waitKey()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
