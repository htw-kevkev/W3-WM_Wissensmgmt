# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 14:10:00 2020

@author: kevin

This script uses tesseract's conf value during ocr processing and writes the metrics to the dedicated directory
Tesseract returns a conf value for every string it finds in every block
Two files are saved: one shows the avg conf value per block (tesseract finds multiple blocks of text in every image)
	and the other file shows the avg conf value per image (avg over all strings which were found)
"""
import pytesseract
from pytesseract import Output
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import csv
import utils
import datetime
import pandas as pd


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
			results_per_block, conf_of_image, blocks_of_image = handle_image(completePath, file, total_images, image_counter)

			# handle information per block
			try: # iterrows will not work if handle_image runs into exception
				for index, row in results_per_block.iterrows():
					blockRow = [directory_name, file, row[0], row[1]]
					imageBlockRows.append(blockRow)
			except:
				blockRow = [directory_name, file, 'ERROR', 'ERROR']
				imageBlockRows.append(blockRow)

			# handle information per image
			imageRow = [directory_name, file, blocks_of_image, conf_of_image]
			imageRows.append(imageRow)

			# log.info(handle_image(completePath, file, total_images, image_counter))
	except Exception as e:
		log.error('There was a problem handling the directory ' + str(directory_name) + ': ' + str(e))


def handle_image(path, file, total_images, image_counter):
	log.info('Start handling image ' + str(image_counter) + ' of ' + str(total_images) + ': ' + str(file))
	
	try:
		image = cv2.imread(path)
		data = pytesseract.image_to_data(image, lang='deu+eng', config=custom_oem_psm_config, output_type=Output.DATAFRAME)

		# remove all rows with no confidence value (-1)
		data = data[data.conf != -1]

		### get confidence and text per block (tesseract detects multiple blocks of text per image)
		# get text per block
		texts_per_block = data.groupby('block_num')['text'].apply(list)
		# get mean confidence value per block
		confs_per_block = data.groupby(['block_num'])['conf'].mean()
		# join texts and confs
		results_per_block = pd.merge(confs_per_block, texts_per_block, left_on='block_num', right_index=True)

		### get one conficence value (mean of all confidence values) and whole text per image
		# get mean confidence value of image
		conf_of_image = data['conf'].mean()
		# get number of blocks
		blocks_of_image = len(results_per_block['conf'])

		log.info('Number of blocks found: ' + str(blocks_of_image))
		log.info('Avg conf value: ' + str(round(conf_of_image,1)))
		log.info('')

		return results_per_block, conf_of_image, blocks_of_image
	except Exception as e:
		log.error('There was a problem handling the image ' + str(file) + ': ' + str(e))
		return 'ERROR', 'ERROR', 'ERROR'


log.info('Inizialize lists for image data')
imageBlockRows = []
imageRows = []

# dirtest = 'C:/Users/kevin/OneDrive/Studium/4_WiSe20_21/1_W3-WM/app_data/test_images'
# dirtest_name = 'test_images'

dir1 = 'C:/Users/kevin/OneDrive/Studium/4_WiSe20_21/1_W3-WM/app_data/Plakatfotos'
dir1_name = 'plakatfotos'

dir2 = 'C:/Users/kevin/OneDrive/Studium/4_WiSe20_21/1_W3-WM/app_data/insta_posters'
dir2_name = 'insta_posters'

# handle_directory(dirtest, dirtest_name)
handle_directory(dir2, dir2_name)
handle_directory(dir1, dir1_name)

file_dir = os.path.dirname(os.path.abspath(__file__))
metrics_folder = 'metrics'
metrics_path = os.path.join(file_dir, metrics_folder)

timestamp = '{:%Y-%m-%d-%H%M%S}'.format(datetime.datetime.now())

# write block information to csv
csvfilename = str(timestamp) + '_metrics_per_block' + '.csv'
csv_path = os.path.join(metrics_path, csvfilename)
log.info('Write list with block data to file ' + str(csvfilename))
columns = ['DIRECTORY', 'FILE', 'CONF', 'TEXT']
with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
	csvwriter = csv.writer(csvfile)
	csvwriter.writerow(columns)
	csvwriter.writerows(imageBlockRows)

# write image information to csv
csvfilename = str(timestamp) + '_metrics_per_image' + '.csv'
csv_path = os.path.join(metrics_path, csvfilename)
log.info('Write list with image data to file ' + str(csvfilename))
columns = ['DIRECTORY', 'FILE', 'BLOCKS', 'CONF']
with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
	csvwriter = csv.writer(csvfile)
	csvwriter.writerow(columns)
	csvwriter.writerows(imageRows)

log.info('All done')