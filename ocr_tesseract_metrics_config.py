# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 14:10:00 2020

@author: kevin

This script uses tesseract's conf value during ocr processing and writes the metrics to the dedicated directory
Tesseract returns a conf value for every string it finds in every block
Three files are saved: one shows the avg conf value per block (tesseract finds multiple blocks of text in every image)
	the second other file shows the avg conf value per image (avg over all strings which were found)
	and the third file shows the avg conf value per configuration of psm and image preprocessing
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
import time
import datetime
import image_preprocessing

# Installation pytesseract ist etwas tricky
# 1. pip install pytesseract in anaconda env
# 2. hier den 64-bit installer herunterladen und ausfuehren: https://github.com/UB-Mannheim/tesseract/wiki
# 3. deutsche Sprache hinzufügen wie hier beschrieben: https://github.com/UB-Mannheim/tesseract/wiki/Install-additional-language-and-script-models
# 4. die Kommandozeile unten im Code ausfuehren
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# initialize logger
log = utils.set_logging()

starttime = '{:%Y-%m-%d-%H%M%S}'.format(datetime.datetime.now())

# function to apply ocr on images
def handle_image(path, file, custom_oem_psm_config, rgb, gray, blur, thres, resc, inv):
	try:
		image = cv2.imread(path)
		starttime_ocr = datetime.datetime.now()

		# IMAGE PREPROCESSING
		if rgb:
			image = image_preprocessing.bgr_to_rgb(image)
		if gray:
			image = image_preprocessing.grayscaling(image)
		if blur:
			image = image_preprocessing.blurring(image)
		if thres:
			image = image_preprocessing.thresholding(image)
		if resc:
			image = image_preprocessing.rescaling(image)
		if inv:
			image = image_preprocessing.inverting(image)

		data = pytesseract.image_to_data(image, lang='deu+eng', config=custom_oem_psm_config, output_type=Output.DATAFRAME)
		endtime_ocr = datetime.datetime.now()
		runtime_ocr = endtime_ocr - starttime_ocr
		runtime_ocr = round(runtime_ocr.total_seconds(),0)

		# remove all rows with no confidence value (-1)...
		data = data[data.conf != -1]
		# ...and empty string as text
		data = data[data.text.replace(' ','') != '']
		data = data[data.text != '    ']

		### get confidence and text per block (tesseract detects multiple blocks of text per image)
		# get text per block
		texts_per_block = data.groupby('block_num')['text'].apply(list)
		# get mean confidence value per block
		confs_per_block = data.groupby(['block_num'])['conf'].mean()
		# join texts and confs
		results_per_block = pd.merge(confs_per_block, texts_per_block, left_on='block_num', right_index=True)

		### get one conficence value (mean of all confidence values) and whole text per image
		# get mean confidence value of image
		conf_of_image = round(data['conf'].mean(),1)
		# get number of blocks
		blocks_of_image = len(results_per_block['conf'])

		# log.info('Number of blocks found: ' + str(blocks_of_image))
		# log.info('Avg conf value: ' + str(conf_of_image))
		# log.info('Runtime: ' + str(runtime_ocr))
		# log.info('')

		# number of found strings has to be returned for aggregations on config level
		no_of_data = len(data['conf'])
		sum_of_conf = data['conf'].sum()

		# in case of everything ran smoothly
		image_status = True

		return image_status, results_per_block, conf_of_image, blocks_of_image, runtime_ocr, no_of_data, sum_of_conf
	except Exception as e:
		log.error('There was a problem handling the image ' + str(file) + ': ' + str(e))
		return False, None, None, None, None, None, None


log.info('Inizialize lists for image data')
imageBlockRows = [] # list to save data on text block level
imageRows = [] # list to save data on image level (aggregated over blocks in image)
configRows = [] # list to save data on config level (aggregated over blocks found during config was in use)


directory = 'C:/Users/kevin/OneDrive/Studium/4_WiSe20_21/1_W3-WM/app_data/test_images'
directory_name = 'test_images'
listOfFiles = os.listdir(directory)
total_images = len(listOfFiles)

# Configuration for tesseract
# https://pypi.org/project/pytesseract/ and https://ai-facets.org/tesseract-ocr-best-practices/
# type 'tesseract --help-oem' or 'tesseract --help-psm' in command prompt to get possible values

# initializing potential oem (OCR Engine modes) values
# oem = 3 is default (based on what is available)
# ==> Realization after testing:
#		only oem=1 makes sense as it is the lstm only mode
#		our fast tessdata is not compatible with legacy and it will not be supported in the future anyway
oem = 1

# initializing potential psm (Page segmentation modes) values
# psm = 3 is default
# skipped modes 4 - 10 are not suitable for our use case and modes 0 and 2 as they raise an error
psm = [1,3,11,12,13]

# initializing variables for image preprocessing steps - using lists to iterate over and eventually use every combination possible
bgr_to_rgb = [False,True]
grayscaling = [False,True]
blurring = [False,True]
thresholding = [False,True]
rescaling = [False,True]
inverting = [False,True]


for p in psm:
	for inv in inverting:
		for resc in rescaling:
			for thres in thresholding:
				for blur in blurring:
					for gray in grayscaling:
						for rgb in bgr_to_rgb:
							custom_oem_psm_config = r'--oem {} --psm {}'.format(oem,p)
							# custom_oem_psm_config = r'--oem 1 --psm 12'
							full_config = '--oem {} --psm {}'.format(oem,p) + ' --rgb {}'.format(rgb) + ' --gray {}'.format(gray) + ' --blur {}'.format(blur) + ' --thres {}'.format(thres) + ' --resc {}'.format(resc) + ' --inv {}'.format(inv)
							log.info('=== Config ' + full_config)

							# initializing variables for avg conf value on config level and image counter for nice logs
							total_data = 0
							total_conf = 0
							total_blocks = 0
							total_runtime = 0
							image_counter_for_log = 0
							success_images = 0
							for file in listOfFiles:
								image_counter_for_log = image_counter_for_log + 1
								completePath = os.path.join(directory, file).replace('\\','/')

								log.info('Start handling image ' + str(image_counter_for_log) + ' of ' + str(total_images) + ': ' + str(file))
								image_status, results_per_block, conf_of_image, blocks_of_image, runtime_ocr, no_of_data, sum_of_conf = handle_image(completePath, file, custom_oem_psm_config, rgb, gray, blur, thres, resc, inv)

								# handle information per block
								try: # iterrows will not work if handle_image runs into exception
									for index, row in results_per_block.iterrows():
										blockRow = [oem, p, rgb, gray, blur, thres, resc, inv, directory_name, file, round(row[0],1), row[1]]
										imageBlockRows.append(blockRow)
								except:
									blockRow = [oem, p, rgb, gray, blur, thres, resc, inv, directory_name, file, None, None]
									imageBlockRows.append(blockRow)

								# handle information per image
								imageRow = [oem, p, rgb, gray, blur, thres, resc, inv, directory_name, file, blocks_of_image, conf_of_image, runtime_ocr]
								imageRows.append(imageRow)

								# add additional data lines and conf values in total variables to avg later on config level
								if image_status:
									total_data += no_of_data
									total_blocks += blocks_of_image
									total_conf += sum_of_conf
									total_runtime += runtime_ocr
									success_images += 1

							if success_images > 0:
								avg_conf = round(total_conf/total_data,1)
								avg_runtime = round(total_runtime/success_images,1)
								configRow = [oem, p, rgb, gray, blur, thres, resc, inv, success_images, total_blocks, avg_conf, avg_runtime]
								configRows.append(configRow)
							else:
								avg_conf = None
								avg_runtime = None
								configRow = [oem, p, rgb, gray, blur, thres, resc, inv, None, None, None, None]
								configRows.append(configRow)

							log.info('=== End of Config ' + full_config)
							log.info('=== Images ' + str(success_images) + ' / Blocks ' + str(total_blocks) + ' / Avg_Conf ' + str(avg_conf) + ' / Avg_Runtime ' + str(avg_runtime))
							log.info('')


file_dir = os.path.dirname(os.path.abspath(__file__))
metrics_folder = 'metrics'
metrics_path = os.path.join(file_dir, metrics_folder)
if not os.path.exists(metrics_path):
    os.makedirs(metrics_path)

# write block information to csv
# csvfilename = str(timestamp) + '_metrics_per_block' + '.csv'
csvfilename = 'metrics_per_block' + '.csv'
csv_path = os.path.join(metrics_path, csvfilename)
log.info('Write list with block data to file ' + str(csvfilename))
columns = ['OEM', 'PSM', 'RGB', 'GRAY', 'BLUR', 'THRES', 'RESC', 'INV', 'DIRECTORY', 'FILE', 'CONF', 'TEXT']
with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
	csvwriter = csv.writer(csvfile)
	csvwriter.writerow(columns)
	csvwriter.writerows(imageBlockRows)

# write image information to csv
# csvfilename = str(timestamp) + '_metrics_per_image' + '.csv'
csvfilename = 'metrics_per_image' + '.csv'
csv_path = os.path.join(metrics_path, csvfilename)
log.info('Write list with image data to file ' + str(csvfilename))
columns = ['OEM', 'PSM', 'RGB', 'GRAY', 'BLUR', 'THRES', 'RESC', 'INV', 'DIRECTORY', 'FILE', 'BLOCKS', 'CONF', 'RUNTIME']
with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
	csvwriter = csv.writer(csvfile)
	csvwriter.writerow(columns)
	csvwriter.writerows(imageRows)

# write config information to csv
# csvfilename = str(timestamp) + '_metrics_per_image' + '.csv'
csvfilename = 'metrics_per_config' + '.csv'
csv_path = os.path.join(metrics_path, csvfilename)
log.info('Write list with config data to file ' + str(csvfilename))
columns = ['OEM', 'PSM', 'RGB', 'GRAY', 'BLUR', 'THRES', 'RESC', 'INV', 'IMAGES', 'BLOCKS', 'CONF', 'RUNTIME']
with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
	csvwriter = csv.writer(csvfile)
	csvwriter.writerow(columns)
	csvwriter.writerows(configRows)


endtime = '{:%Y-%m-%d-%H%M%S}'.format(datetime.datetime.now())
log.info('All done / ' + 'started ' + str(starttime) + ' / finished ' + str(endtime))