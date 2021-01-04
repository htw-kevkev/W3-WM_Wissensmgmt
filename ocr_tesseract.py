# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 13:49:26 2020

@author: kevin

This file uses Tesseract-OCR to detect text in images and writes the text per image to csv
"""
import pytesseract
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import csv
import utils
import image_preprocessing


# Installation pytesseract ist etwas tricky
# 1. pip install pytesseract in anaconda env
# 2. hier den 64-bit installer herunterladen und ausfuehren: https://github.com/UB-Mannheim/tesseract/wiki
# 3. deutsche Sprache hinzufügen wie hier beschrieben: https://github.com/UB-Mannheim/tesseract/wiki/Install-additional-language-and-script-models
# 4. die Kommandozeile unten im Code ausfuehren
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configuration for tesseract
# https://pypi.org/project/pytesseract/ and https://ai-facets.org/tesseract-ocr-best-practices/
custom_oem_psm_config = r'--oem 1 --psm 12'

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
			ocr_after_rgb, ocr_after_grayscaling, ocr_after_blurring = handle_image(completePath, file, total_images, image_counter)
			row = [directory_name, file, ocr_after_rgb, ocr_after_grayscaling, ocr_after_blurring]
			imageRows.append(row)
	except:
		log.error('There was a problem handling the directory ' + str(directory_name))


def handle_image(path, file, total_images, image_counter):
	log.info('Start handling image ' + str(image_counter) + ' of ' + str(total_images) + ': ' + str(file))
	
	try:
		image = cv2.imread(path)
		image = image_preprocessing.bgr_to_rgb(image)
		ocr_after_rgb = pytesseract.image_to_string(image, lang='deu+eng', config=custom_oem_psm_config).replace(',','').replace(';','').replace('\n', ' ')

		image = image_preprocessing.grayscaling(image)
		ocr_after_grayscaling = pytesseract.image_to_string(image, lang='deu+eng', config=custom_oem_psm_config).replace(',','').replace(';','').replace('\n', ' ')

		image = image_preprocessing.blurring(image)
		ocr_after_blurring = pytesseract.image_to_string(image, lang='deu+eng', config=custom_oem_psm_config).replace(',','').replace(';','').replace('\n', ' ')

		return ocr_after_rgb, ocr_after_grayscaling, ocr_after_blurring
	except:
		log.error('There was a problem handling the image ' + str(file))
		return 'ERROR', 'ERROR', 'ERROR'


log.info('Inizialize list for image data')
imageRows = []

dirtest = 'C:/Users/kevin/OneDrive/Studium/4_WiSe20_21/1_W3-WM/app_data/test_images'
dirtest_name = 'test_images'

# dir1 = 'C:/Users/kevin/OneDrive/Studium/4_WiSe20_21/1_W3-WM/app_data/Plakatfotos'
# dir1_name = 'plakatfotos'

# dir2 = 'C:/Users/kevin/OneDrive/Studium/4_WiSe20_21/1_W3-WM/app_data/insta_posters'
# dir2_name = 'insta_posters'

handle_directory(dirtest, dirtest_name)
# handle_directory(dir1, dir1_name)
# handle_directory(dir2, dir2_name)

csvfilename = 'images_to_text.csv'
log.info('Write list with image data to file ' + str(csvfilename))
columns = ['DIRECTORY', 'FILE', 'ocr_after_rgb', 'ocr_after_grayscaling', 'ocr_after_blurring']
with open(csvfilename, 'w', encoding='utf-8', newline='') as csvfile:
	csvwriter = csv.writer(csvfile)
	csvwriter.writerow(columns)
	csvwriter.writerows(imageRows)

log.info('All done')











