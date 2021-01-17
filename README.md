# Student project for text recognition within poster images using Tesseract OCR
Hochschule für Technik und Wirtschaft Berlin

# Authors
- Kevin Kretzschmar
- Lucas Pohling

# References
- [Tesseract][tesseract-ocr]
- [Tesseract documentation][Tessdoc]
- [Tesseract at UB Mannheim][UB]

# Requirements
- Installation of Tesseract (for Windows) - we used the installer of UB Mannheim
- Python 3.7 or higher
- Pytesseract library to use the API
- OpenCV library for image preprocessing

# Files and purpose
## ocr_tesseract.py
Basic file which takes images stored in a defined directory, does basic image preprocessing steps, uses Tesseract-OCR to detect text in images and writes the text per image to csv

## ocr_tesseract_metrics_config.py
This script uses tesseract's conf value during ocr processing and writes the metrics to the dedicated directory. Tesseract returns a conf value for every string it finds in every block.
Three files are saved:
1. Avg conf value per block (tesseract finds multiple blocks of text in every image)
2. Avg conf value per image (avg over all strings which were found)
3. Avg conf value per configuration of psm and image preprocessing
Based on the conf value, decisions can be made on what is the ideal psm and oem config and what are the ideal image preprocessing steps for the images used.

## ocr_tesseract_metrics_lang.py
This script is very similar to the latter one but its purpose is not to find out the best config but the best combination of Tesseract's lang files. This is especially useful when a custom lang data was trained and should be compared to the standard files. All lang files used have to be available in the dedicated Tesseract installation directory and have to be compatible with the installed Tesseract version.

## image_preprocessing.py
This script contains functions for image preprocessing via OpenCV library before ocr and is used in the latter scripts.

## ocr_google.py
This script uses Google Vision API to detect text in images and writes the text per image to csv. Credentials for the API are necessary! We used this script to compare how Google API performed compared to Tesseract on our images.

## neocr_conversion.py
This script is to convert the NEOCR dataset (with jpg and xml files) to Tesseract friendly format (tif and box files). Makes it usable for training.
The dataset can be found here: [NEOCR][NEOCR]

[tesseract-ocr]: https://github.com/tesseract-ocr/tesseract
[Tessdoc]: https://tesseract-ocr.github.io/
[UB]: https://github.com/UB-Mannheim/tesseract/wiki
[NEOCR]: http://www.iapr-tc11.org/mediawiki/index.php?title=NEOCR:_Natural_Environment_OCR_Dataset
