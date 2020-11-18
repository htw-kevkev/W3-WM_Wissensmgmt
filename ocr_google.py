
from google.cloud import vision
import io
import os
import cv2
import matplotlib.pyplot as plt

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="credentials.json"

# from https://cloud.google.com/vision/docs/ocr#vision_text_detection-python
def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')

    for text in texts:
        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

        print('bounds: {}'.format(','.join(vertices)))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))


def show_image(image):
    plt.imshow(image)
    # plt.title(file)
    plt.show()


dir_test = 'C:/Users/kevin/OneDrive/Studium/4_WiSe20_21/1_W3-WM/app_data/test_images'

listOfFiles = os.listdir(dir_test)
for file in listOfFiles:
    print('== Processing file ' + str(file) + ' ==')
    path = os.path.join(dir_test, file).replace('\\','/')

    detect_text(path)

    image = cv2.imread(completePath, cv2.IMREAD_COLOR)
    show_image(image)

