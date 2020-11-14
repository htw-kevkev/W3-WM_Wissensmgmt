# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 13:49:26 2020

@author: kevin
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

# initialize logger
log = utils.set_logging()


def handle_directory(directory, directory_name):
	log.info('Start handling directory ' + str(directory_name))
	
	try:
		listOfFiles = os.listdir(directory)
		total_images = len(listOfFiles)
		image_counter = 0
		for file in listOfFiles:
			image_counter = image_counter + 1
			completePath = os.path.join(directory, file).replace('\\','/')
			ocr_before_preprocessing, ocr_after_grayscaling, ocr_after_rescaling, ocr_after_blurring, ocr_after_thresholding = handle_image(completePath, file, total_images, image_counter)
			row = [directory_name, file, ocr_before_preprocessing, ocr_after_grayscaling, ocr_after_rescaling, ocr_after_blurring, ocr_after_thresholding]
			imageRows.append(row)
	except:
		log.error('There was a problem handling the directory ' + str(directory_name))


def handle_image(path, file, total_images, image_counter):
	log.info('Start handling image ' + str(image_counter) + ' of ' + str(total_images) + ': ' + str(file))
	
	try:
		image = cv2.imread(path)

		# need to replace commas and semicolons as this is our separator for csv file and also remove line breaks
		ocr_before_preprocessing = pytesseract.image_to_string(image, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')

		# === Image preprocessing ===
		# https://www.freecodecamp.org/news/getting-started-with-tesseract-part-ii-f7f9a0899b3f/
		# https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html

		# invert image
		# image = cv2.bitwise_not(image)
		# ocr_after_inverting = pytesseract.image_to_string(image, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')

		# image to grayscale
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		ocr_after_grayscaling = pytesseract.image_to_string(image, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')

		# == A. Rescaling
		# 1. shrinking image
		image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
		# 2. enlarge image
		# image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
		ocr_after_rescaling = pytesseract.image_to_string(image, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')

		# === B. Blurring
		# 1. Averaging
		# image = cv.blur(image,(5,5))
		# 2. 2. Gaussian blurring
		image = cv2.GaussianBlur(image, (5, 5), 0)
		# 3. Median blurring
		# image = cv2.medianBlur(image, 3)
		# 4. Bilateral filtering
		# image = cv2.bilateralFilter(image,9,75,75)
		ocr_after_blurring = pytesseract.image_to_string(image, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')

		# === C. Image Thresholding
		# 1. Simple Threshold
		# thr_value, image = cv2.threshold(image,127,255,cv.THRESH_BINARY)
		# 2. Adaptive Threshold
		# image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
		# 3. Otsu’s Threshold
		thr_value, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		ocr_after_thresholding = pytesseract.image_to_string(image, lang='deu+eng').replace(',','').replace(';','').replace('\n', ' ')

		# # show preprocessed image (from https://stackoverflow.com/questions/2659312/how-do-i-convert-a-numpy-array-to-and-display-an-image)
		# plt.imshow(image)
		# plt.title(file)
		# plt.show()

		return ocr_before_preprocessing, ocr_after_grayscaling, ocr_after_rescaling, ocr_after_blurring, ocr_after_thresholding
	except:
		log.error('There was a problem handling the image ' + str(file))
		return 'ERROR', 'ERROR', 'ERROR', 'ERROR', 'ERROR'


log.info('Inizialize list for image data')
imageRows = []

# dirtest = 'C:/Users/kevin/OneDrive/Studium/4_WiSe20_21/1_W3-WM/app_data/test_images'
# dirtest_name = 'test_images'

dir1 = 'C:/Users/kevin/OneDrive/Studium/4_WiSe20_21/1_W3-WM/app_data/Plakatfotos'
dir1_name = 'plakatfotos'

dir2 = 'C:/Users/kevin/OneDrive/Studium/4_WiSe20_21/1_W3-WM/app_data/insta_posters'
dir2_name = 'insta_posters'

# handle_directory(dirtest, dirtest_name)
handle_directory(dir1, dir1_name)
handle_directory(dir2, dir2_name)

csvfilename = 'images_to_text.csv'
log.info('Write list with image data to file ' + str(csvfilename))
columns = ['DIRECTORY', 'FILE', 'ocr_before_preprocessing', 'ocr_after_grayscaling', 'ocr_after_rescaling', 'ocr_after_blurring', 'ocr_after_thresholding']
with open(csvfilename, 'w', encoding='utf-8', newline='') as csvfile:
	csvwriter = csv.writer(csvfile)
	csvwriter.writerow(columns)
	csvwriter.writerows(imageRows)

log.info('All done')











