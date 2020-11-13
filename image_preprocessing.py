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

dir_test = 'C:/Users/kevin/OneDrive/Studium/4_WiSe20_21/1_W3-WM/app_data/test_images'

listOfFiles = os.listdir(dir_test)
for file in listOfFiles:
	print('===== Processing file ' + str(file) + ' =====')

	completePath = os.path.join(dir_test, file).replace('\\','/')
	image = cv2.imread(completePath)
	z = pytesseract.image_to_string(image, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')
	print('== Z: ' + z)

	# === Image preprocessing ===
	# https://www.freecodecamp.org/news/getting-started-with-tesseract-part-ii-f7f9a0899b3f/
	# https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html

	# invert image
	# image = cv2.bitwise_not(image)
	# i = pytesseract.image_to_string(image, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')
	# print('== I: ' + i)

	# image to grayscale
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	g = pytesseract.image_to_string(image, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')
	print('== G: ' + g)

	# == A. Rescaling
	# 1. shrinking image
	# image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
	# 2. enlarge image
	image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
	r = pytesseract.image_to_string(image, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')
	print('== R: ' + r)

	# === B. Blurring
	# 1. Averaging
	# image = cv.blur(image,(5,5))
	# 2. 2. Gaussian blurring
	image = cv2.GaussianBlur(image, (5, 5), 0)
	# 3. Median blurring
	# image = cv2.medianBlur(image, 3)
	# 4. Bilateral filtering
	# image = cv2.bilateralFilter(image,9,75,75)
	b = pytesseract.image_to_string(image, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')
	print('== B: ' + b)

	# === C. Image Thresholding
	# 1. Simple Threshold
	# thr_value, image = cv2.threshold(image,127,255,cv.THRESH_BINARY)
	# 2. Adaptive Threshold
	# image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
	# 3. Otsu’s Threshold
	thr_value, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	t = pytesseract.image_to_string(image, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')
	print('== T: ' + t)

	# show preprocessed image (from https://stackoverflow.com/questions/2659312/how-do-i-convert-a-numpy-array-to-and-display-an-image)
	# plt.imshow(image)
	# plt.title(file)
	# plt.show()
	print('\n')

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

	









