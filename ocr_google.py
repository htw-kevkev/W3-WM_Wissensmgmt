
# !!! INTERESSANT: https://www.tarent.de/blog/tesseract-opencvs-east-oder-doch-lieber-google-vision-api

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 17:26:00 2020

@author: kevin
"""
from google.cloud import vision
import io
import os
import cv2
import matplotlib.pyplot as plt
import csv
import utils
# LUCAS LUCASAAAAAAAAAAAAAAAA LUCAS
#LUUUUUUUUUCAAAAAAAAAAAAS
# initialize logger
log = utils.set_logging()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="credentials.json"


def handle_directory(directory, directory_name):
    log.info('Start handling directory ' + str(directory_name))
    
    try:
        listOfFiles = os.listdir(directory)
        total_images = len(listOfFiles)
        image_counter = 0
        for file in listOfFiles:
            image_counter = image_counter + 1
            path = os.path.join(directory, file).replace('\\','/')
            ocr_google = detect_text(path, file, total_images, image_counter)
            row = [directory_name, file, ocr_google]
            imageRows.append(row)
    except:
        log.error('There was a problem handling the directory ' + str(directory_name))


# from https://cloud.google.com/vision/docs/ocr#vision_text_detection-python
def detect_text(path, file, total_images, image_counter):
    log.info('Start handling image ' + str(image_counter) + ' of ' + str(total_images) + ': ' + str(file))

    try:
        """Detects text in the file."""
        client = vision.ImageAnnotatorClient()

        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        response = client.text_detection(image=image)
        texts = response.text_annotations
        # print('Texts:')

        # first description already contains all text elements
        ocr_google = texts[0].description
        ocr_google = ocr_google.replace(',','').replace(';','').replace('|','').replace('\n', '|')

        # for text in texts:
        #     print(text.description)

            # print('\n"{}"'.format(text.description))

            # vertices = (['({},{})'.format(vertex.x, vertex.y)
            #             for vertex in text.bounding_poly.vertices])

            # print('bounds: {}'.format(','.join(vertices)))

        # if response.error.message:
        #     raise Exception(
        #         '{}\nFor more info on error messages, check: '
        #         'https://cloud.google.com/apis/design/errors'.format(
        #             response.error.message))

        return ocr_google
    except:
        log.error('There was a problem handling the image ' + str(file))
        return 'ERROR'

def show_image(image):
    plt.imshow(image)
    # plt.title(file)
    plt.show()


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
columns = ['DIRECTORY', 'FILE', 'ocr_google']
with open(csvfilename, 'w', encoding='utf-8', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(columns)
    csvwriter.writerows(imageRows)

log.info('All done')

