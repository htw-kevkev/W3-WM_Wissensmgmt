"""
Created on Mon Nov 13 13:52:00 2020

@author: kevin

This file is to test optimal image preprocessing for ocr
"""
import pytesseract
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import csv
import utils

# Installation pytesseract ist etwas tricky
# 1. pip install pytesseract in anaconda env
# 2. hier den 64-bit installer herunterladen und ausfuehren: https://github.com/UB-Mannheim/tesseract/wiki
# 3. deutsche Sprache hinzufügen wie hier beschrieben: https://github.com/UB-Mannheim/tesseract/wiki/Install-additional-language-and-script-models
# 4. die Kommandozeile unten im Code ausfuehren
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# INTERESTING REFERENCES
# https://ai-facets.org/tesseract-ocr-best-practices/
# https://www.statworx.com/de/blog/finetuning-von-tesseract-ocr-fuer-deutsche-rechnungen/

# Configuration for tesseract
# https://pypi.org/project/pytesseract/ and https://ai-facets.org/tesseract-ocr-best-practices/
custom_oem_psm_config = r'--oem 1 --psm 12'


def show_image(image):
	plt.imshow(image)
	# plt.title(file)
	plt.show()

def apply_laplacian(img):

    # [variables]
    # Declare the variables we are going to use
    ddepth = cv2.CV_16S
    kernel_size = 3
    # [variables]
    # [load]
    src = img # Load an image
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


dir_test = 'C:/Users/kevin/OneDrive/Studium/4_WiSe20_21/1_W3-WM/app_data/test_images'

listOfFiles = os.listdir(dir_test)
for file in listOfFiles:
	print('== Processing file ' + str(file) + ' ==')
	completePath = os.path.join(dir_test, file).replace('\\','/')
	image = cv2.imread(completePath, cv2.IMREAD_COLOR)

	# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	image = apply_laplacian(image)

	text = pytesseract.image_to_string(image, lang='deu+eng', config=custom_oem_psm_config).replace(',','').replace(';','').replace('\n', ' ')
	print(text)

	show_image(image)

	'''
	print('== Processing file ' + str(file) + ' ==')


	# === OCR w/o preprocessing
	z = pytesseract.image_to_string(image, lang='deu+eng', config=custom_oem_psm_config).replace(',','').replace(';','').replace('\n', ' ')
	print('== 0: ' + z)
	# show_image(image, file + '\nbefore preprocessing')


	# === Simple image preprocessing approach ===
	# https://pypi.org/project/pytesseract/

	# By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
	# we need to convert from BGR to RGB format/mode:
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	one = pytesseract.image_to_string(image, lang='deu+eng', config=custom_oem_psm_config).replace(',','').replace(';','').replace('\n', ' ')
	print('== 1: ' + one)
	# show_image(image, file + '\nRGB')
	# OR --- THIS IS SUPERLOW IN PERFORMANCE
	# img_rgb2 = Image.frombytes('RGB', image.shape[:2], image, 'raw', 'BGR', 0, 0)
	# two = pytesseract.image_to_string(img_rgb2, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')

	image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	two = pytesseract.image_to_string(image, lang='deu+eng', config=custom_oem_psm_config).replace(',','').replace(';','').replace('\n', ' ')
	print('== 2: ' + two)
	# show_image(image, file + '\nGRAY')

	image = cv2.medianBlur(image,5)
	three = pytesseract.image_to_string(image, lang='deu+eng', config=custom_oem_psm_config).replace(',','').replace(';','').replace('\n', ' ')
	print('== 3: ' + three)
	# show_image(image, file + '\nBLUR')

	image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
	four = pytesseract.image_to_string(image, lang='deu+eng', config=custom_oem_psm_config).replace(',','').replace(';','').replace('\n', ' ')
	print('== 4: ' + four)
	# show_image(image, file + '\nTHRESHOLD')


	print('\n')
	'''

	
	# # === Image preprocessing ===
	# # https://nanonets.com/blog/ocr-with-tesseract/#preprocessingfortesseract

	# # get grayscale image
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


	# # noise removal
	# image = cv2.medianBlur(image,5)


	# # thresholding
	# image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

	# t = pytesseract.image_to_string(image, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')
	# print('== T: ' + t)
	# show_image(image, file + '\nafter thresholding')


	# # dilation (Streckung)
	# kernel = np.ones((5,5),np.uint8)
	# image = cv2.dilate(image, kernel, iterations = 1)


 #    # erosion
	# kernel = np.ones((5,5),np.uint8)
	# image = cv2.erode(image, kernel, iterations = 1)


 #    # opening - erosion followed by dilation
	# kernel = np.ones((5,5),np.uint8)
	# image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


 #    # canny edge detection
	# image = cv2.Canny(image, 100, 200)


 #    # skew correction
	# coords = np.column_stack(np.where(image > 0))
	# angle = cv2.minAreaRect(coords)[-1]
	# if angle < -45:
	# 	angle = -(90 + angle)
	# else:
	# 	angle = -angle
	# (h, w) = image.shape[:2]
	# center = (w // 2, h // 2)
	# M = cv2.getRotationMatrix2D(center, angle, 1.0)
	# image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


	# e = pytesseract.image_to_string(image, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')
	# print('== E: ' + e)
	# show_image(image, file + '\nafter everything')



	# === Image preprocessing ===
	# https://www.freecodecamp.org/news/getting-started-with-tesseract-part-ii-f7f9a0899b3f/
	# https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html

	# invert image
	# image = cv2.bitwise_not(image)
	# i = pytesseract.image_to_string(image, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')
	# print('== I: ' + i)

	# image to grayscale
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# g = pytesseract.image_to_string(image, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')
	# print('== G: ' + g)
	# plt.imshow(image)
	# plt.title(file + 'gray')
	# plt.show()


	# == A. Rescaling
	# 1. shrinking image
	# image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
	# 2. enlarge image
	# image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
	# r = pytesseract.image_to_string(image, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')
	# print('== R: ' + r)
	# plt.imshow(image)
	# plt.title(file + 'rescaled')
	# plt.show()


	# === B. Blurring
	# 1. Averaging
	# image = cv.blur(image,(5,5))
	# 2. 2. Gaussian blurring
	# image = cv2.GaussianBlur(image, (5, 5), 0)
	# 3. Median blurring
	# image = cv2.medianBlur(image, 3)
	# 4. Bilateral filtering
	# image = cv2.bilateralFilter(image,9,75,75)
	# b = pytesseract.image_to_string(image, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')
	# print('== B: ' + b)
	# plt.imshow(image)
	# plt.title(file + 'blurred')
	# plt.show()


	# === C. Image Thresholding
	# 1. Simple Threshold
	# thr_value, image = cv2.threshold(image,127,255,cv.THRESH_BINARY)
	# 2. Adaptive Threshold
	# image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
	# 3. Otsu’s Threshold
	# thr_value, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	# t = pytesseract.image_to_string(image, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')
	# print('== T: ' + t)
	# plt.imshow(image)
	# plt.title(file + 'thresholded')
	# plt.show()


	# show preprocessed image (from https://stackoverflow.com/questions/2659312/how-do-i-convert-a-numpy-array-to-and-display-an-image)
	# plt.imshow(image)
	# plt.title(file)
	# plt.show()

	# === OLD ===

	# no_preprocessing = pytesseract.image_to_string(image, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')
	# print('== No preprocessing: ' + no_preprocessing)

	# # invert image
	# image_inv = cv2.bitwise_not(image)
	# after_inv = pytesseract.image_to_string(image_inv, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')
	# print('== After inverting: ' + after_inv)

	# # image to grayscale
	# image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# after_gray = pytesseract.image_to_string(image_gray, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')
	# print('== After grayscaling: ' + after_gray)

	# # invert image and grayscale
	# image_inv_gray = cv2.cvtColor(cv2.bitwise_not(image), cv2.COLOR_BGR2GRAY)
	# after_inverting_and_grayscaling = pytesseract.image_to_string(image_inv_gray, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')
	# print('== After inverting and grayscaling: ' + after_inverting_and_grayscaling)

	### thresholding (from https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html)
	# Adaptive Gaussian Thresholding - suuuper inperfomant und bringt keine Ergebnisse....
	# img1 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
 #            cv2.THRESH_BINARY,11,2)
	# adaptive_gaussian = pytesseract.image_to_string(img1, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')
	# print('Gaussian:')
	# print(adaptive_gaussian)

	# Otsu's thresholding after Gaussian filtering
	# blur = cv2.GaussianBlur(image,(5,5),0)
	# ret,img2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# otsu = pytesseract.image_to_string(img2, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')
	# print('== After tresholding:')
	# print(otsu)

	# need to replace commas and semicolons as this is our separator for csv file and also remove line breaks
	# string_after_noise_elimination = pytesseract.image_to_string(image, lang='eng+deu').replace(',','').replace(';','').replace('\n', ' ')
	# string_after_noise_elimination_deu = pytesseract.image_to_string(image, lang='deu').replace(',','').replace(';','').replace('\n', ' ')
	# print(string_after_noise_elimination)
	# print(string_after_noise_elimination_deu)

	









