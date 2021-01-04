# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:52:00 2020

@author: kevin

This file contains functions for image preprocessing before ocr

REFERENCES
https://ai-facets.org/tesseract-ocr-best-practices/
https://www.statworx.com/de/blog/finetuning-von-tesseract-ocr-fuer-deutsche-rechnungen/
https://pypi.org/project/pytesseract/
https://nanonets.com/blog/ocr-with-tesseract/#preprocessingfortesseract
https://www.freecodecamp.org/news/getting-started-with-tesseract-part-ii-f7f9a0899b3f/
https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
"""
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

def show_image(image):
	plt.imshow(image)
	# plt.title(file)
	plt.show()

# By default OpenCV stores images in BGR format and since pytesseract assumes RGB format, we can convert from BGR to RGB format/mode:
def bgr_to_rgb(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	return image

def grayscaling(image):
	image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	return image

def blurring(image):
	# image = cv.blur(image,(5,5)) # 1. Averaging
	image = cv2.GaussianBlur(image, (5, 5), 0) # 2. Gaussian blurring
	# image = cv2.medianBlur(image, 3) # 3. Median blurring
	# image = cv2.bilateralFilter(image,9,75,75) 4. Bilateral filtering
	return image

def thresholding(image):
	# thr_value, image = cv2.threshold(image,127,255,cv.THRESH_BINARY) # 1. Simple Threshold
	# image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2) # 2. Adaptive Threshold
	thr_value, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # 3. Otsu’s Threshold
	return image

def rescaling(image):
	# 1. shrink image
	# image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
	# 2. enlarge image
	image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
	return image

def inverting(image):
	image = cv2.bitwise_not(image)
	return image

# strecken
def dilation(image):
	kernel = np.ones((5,5),np.uint8)
	image = cv2.dilate(image, kernel, iterations = 1)
	return image

def erosion(image):
	kernel = np.ones((5,5),np.uint8)
	image = cv2.erode(image, kernel, iterations = 1)
	return image

def erosion_dilation(image):
	kernel = np.ones((5,5),np.uint8)
	image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
	return image

def canny(image):
	image = cv2.Canny(image, 100, 200)
	return image

def skew_correction(image):
	coords = np.column_stack(np.where(image > 0))
	angle = cv2.minAreaRect(coords)[-1]
	if angle < -45:
		angle = -(90 + angle)
	else:
		angle = -angle
	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	return image

def apply_laplacian(image):
    # [variables]
    # Declare the variables we are going to use
    ddepth = cv2.CV_16S
    kernel_size = 3
    # [variables]
    # [load]
    src = image # Load an image
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image')
        return -1
    # [load]
    # [reduce_noise]
    # Remove noise by blurring with a Gaussian filter
    src = cv2.GaussianBlur(src, (3, 3), 0)
    # [reduce_noise]
    # [convert_to_gray]
    # Convert the image to grayscale
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # [convert_to_gray]
    # Create Window
    # [laplacian]
    # Apply Laplace function
    dst = cv2.Laplacian(src_gray, ddepth, ksize=kernel_size)
    # [laplacian]
    # [convert]
    # converting back to uint8
    abs_dst = cv2.convertScaleAbs(dst)
    # [convert]
    # [display]
    return abs_dst