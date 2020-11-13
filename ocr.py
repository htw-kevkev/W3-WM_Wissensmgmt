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
		for file in listOfFiles:
			completePath = os.path.join(directory, file).replace('\\','/')
			string_pure, string_pure_deu, string_after_noise_elimination, string_after_noise_elimination_deu = handle_image(completePath, file)
			row = [directory_name, file, string_pure, string_pure_deu, string_after_noise_elimination, string_after_noise_elimination_deu]
			imageRows.append(row)
	except:
		log.error('There was a problem handling the directory ' + str(directory_name))


def handle_image(path, file):
	log.info('Start handling image ' + str(file))
	
	try:
		image = cv2.imread(path)

		# need to replace commas and semicolons as this is our separator for csv file and also remove line breaks
		string_pure = pytesseract.image_to_string(image, lang='eng+deu').replace(',','').replace(';','').replace('\n', ' ')
		string_pure_deu = pytesseract.image_to_string(image, lang='deu').replace(',','').replace(';','').replace('\n', ' ')

		# # eliminate noise in image (from https://towardsdatascience.com/create-simple-optical-character-recognition-ocr-with-python-6d90adb82bb8)
		norm_img = np.zeros((image.shape[0], image.shape[1]))
		image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
		image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)[1]
		image = cv2.GaussianBlur(image, (1, 1), 0)

		# # show preprocessed image (from https://stackoverflow.com/questions/2659312/how-do-i-convert-a-numpy-array-to-and-display-an-image)
		# plt.imshow(image, interpolation='nearest')
		# plt.show()

		# need to replace commas and semicolons as this is our separator for csv file and also remove line breaks
		string_after_noise_elimination = pytesseract.image_to_string(image, lang='eng+deu').replace(',','').replace(';','').replace('\n', ' ')
		string_after_noise_elimination_deu = pytesseract.image_to_string(image, lang='deu').replace(',','').replace(';','').replace('\n', ' ')

		return string_pure, string_pure_deu, string_after_noise_elimination, string_after_noise_elimination_deu
	except:
		log.error('There was a problem handling the image ' + str(file))
		return 'ERROR', 'ERROR', 'ERROR', 'ERROR'


log.info('Inizialize list for image data')
imageRows = []

dir1 = 'C:/Users/kevin/OneDrive/Studium/4_WiSe20_21/1_W3-WM/app_data/Plakatfotos'
dir1_name = 'plakatfotos'

dir2 = 'C:/Users/kevin/OneDrive/Studium/4_WiSe20_21/1_W3-WM/app_data/insta_posters'
dir2_name = 'insta_posters'

handle_directory(dir1, dir1_name)
handle_directory(dir2, dir2_name)

csvfilename = 'images_to_text.csv'
log.info('Write list with image data to file ' + str(csvfilename))
columns = ['DIRECTORY', 'FILE', 'STRING_PURE', 'STRING_PURE_DEU', 'STRING_AFTER_NOISE_ELIMINATION', 'STRING_AFTER_NOISE_ELIMINATION_DEU']
with open(csvfilename, 'w', encoding='utf-8', newline='') as csvfile:
	csvwriter = csv.writer(csvfile)
	csvwriter.writerow(columns)
	csvwriter.writerows(imageRows)

log.info('All done')











