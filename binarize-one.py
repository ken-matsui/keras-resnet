# coding: utf-8

import cv2
import numpy as np
import sys

args = sys.argv


def binarize(img):
    green = img[:, :, 1]
    red = img[:, :, 2]
    redGreen = cv2.addWeighted(red, 0.5, green, 0.5, 0)
    # binalize
    th_red = cv2.adaptiveThreshold(
        redGreen,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        9,
        10)
    # cleaning noise by opening
    kernel = np.ones((1, 1), np.uint8)  # [[1]]
    th_red = cv2.morphologyEx(th_red, cv2.MORPH_OPEN, kernel)

    return th_red


def main():
    img = cv2.imread(args[1], cv2.IMREAD_COLOR)
    b_img = binarize(img)
    cv2.imwrite('b-' + args[1], b_img)


if __name__ == '__main__':
    main()
