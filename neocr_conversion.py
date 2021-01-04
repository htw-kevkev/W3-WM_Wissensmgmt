# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 17:35:00 2020

@author: kevin

This file is to convert the NEOCR dataset (with jpg and xml files) to Tesseract friendly format (tif and box files)
Makes it usable for training
"""
import utils
import os
import sys
import cv2
from lxml import etree

# initialize logger
log = utils.set_logging()

dir_neocr = 'C:/Users/kevin/OneDrive/Studium/4_WiSe20_21/1_W3-WM/app_data/NEOCR/neocr_smallset_kevin'

log.info('Start handling directory ' + str(dir_neocr))

listOfFiles = os.listdir(dir_neocr)
total_images = len(listOfFiles)
image_counter = 0

script_dir = os.path.dirname(os.path.abspath(__file__))
target_path = os.path.join(script_dir, 'NEOCR_TIF_BOX')

for file in listOfFiles:
	image_counter = image_counter + 1
	log.info('Start handling image ' + str(image_counter) + ' of ' + str(total_images) + ': ' + str(file))

	try:
		file_path = os.path.join(dir_neocr, file)
	
		# handle jpg files
		if file[-3:] == 'jpg':
			image = cv2.imread(file_path)
			filename_tif = file[0:-4] + '.tif'
			cv2.imwrite(os.path.join(target_path, filename_tif), image)

		# handle xml files
		if file[-3:] == 'xml':
			tree = etree.parse(file_path)
			root = tree.getroot()
			for child in root.findall('properties'):
				for x in child.findall('height'):
					height = int(x.text)

			xml_data = ''
			for obj in root.findall('object'):
				polygon = obj.find('polygon')
				x_values = []
				y_values = []
				for pt in polygon.findall('pt'):
					x_values.append(int(pt.find('x').text))
					y_values.append(int(pt.find('y').text))

				symbol = obj.find('text').text.replace('<br/>', ';').replace(' ', ';')
				left = x_values[0]
				bottom = height - y_values[2]
				right = x_values[1]
				top = height - y_values[0]
				page = 0

				xml_data = xml_data + symbol + ' ' + str(left) + ' ' + str(bottom) + ' ' + str(right) + ' ' + str(top) + ' ' + str(page) + '\n' 

			filename_box = file[0:-4] + '.box'
			box_file = open(os.path.join(target_path, filename_box), 'w', encoding='utf-8')
			# box_file = open(filename_box, 'w', encoding='utf-8')
			box_file.write(xml_data)
			box_file.close()
	except Exception as e:
		log.error('There was a problem handling the image ' + str(file) + ': ' + str(e))