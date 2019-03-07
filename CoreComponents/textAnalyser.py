
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import time

pytesseract.pytesseract.tesseract_cmd = r"../Tesseract/tesseract"

if __name__ == "__main__":
    image = Image.open("./Assets/185.png")
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(warped, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    print(image)

    last_time = time.time()
    text = pytesseract.image_to_string(image)
    print('Analyse took {} seconds'.format(time.time()-last_time))
    print(text)

    #cv2.imshow("image", image)
    #cv2.waitKey(1)