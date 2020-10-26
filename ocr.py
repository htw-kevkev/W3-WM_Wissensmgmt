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


# Installation pytesseract ist etwas tricky
# 1. pip install pytesseract in anaconda env
# 2. hier den 64-bit installer herunterladen und ausfuehren: https://github.com/UB-Mannheim/tesseract/wiki
# 3. die Kommandozeile unten im Code ausfuehren
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# get filename of image
# filename = "plakat1.png"
filename = "jazzfest.jpg"

# read the image using OpenCV
image = cv2.imread(filename)

# eliminate noise in image (from https://towardsdatascience.com/create-simple-optical-character-recognition-ocr-with-python-6d90adb82bb8)
norm_img = np.zeros((image.shape[0], image.shape[1]))
image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)[1]
image = cv2.GaussianBlur(image, (1, 1), 0)

# show preprocessed image (from https://stackoverflow.com/questions/2659312/how-do-i-convert-a-numpy-array-to-and-display-an-image)
plt.imshow(image, interpolation='nearest')
plt.show()

# get the string
string = pytesseract.image_to_string(image)
print(string[0:100])











