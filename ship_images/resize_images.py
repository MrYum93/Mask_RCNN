#!/usr/bin/env python3

import cv2 as cv
import sys
import os
import glob

jpg_files = []
dir = ""

for roots, dirs, files in os.walk('.'):
    for file in files:
        if "jpg" in file:
            jpg_files.append(file)
            print(file)
            img = cv.imread(file)

            re_img = cv.resize(img, (640, 400))

            saved_filename = "resized_" + file
            cv.imwrite(saved_filename, re_img)
            # cv.imshow("win1", re_img)
            # cv.waitKey(0)



