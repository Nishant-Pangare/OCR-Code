# USAGE
# python ocr_and_spellcheck.py --image comic_spelling.png

# import the necessary packages
from spellchecker import SpellChecker
import pytesseract
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image to be OCR'd")
args = vars(ap.parse_args())

# load the input image and convert it from BGR to RGB channel ordering
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# use Tesseract to OCR the image
text = pytesseract.image_to_string(rgb)

# show the text *before* ocr-spellchecking has been applied
print("BEFORE SPELLCHECK")
print("=================")
print(text)
print("\n")

# apply spell checking to the OCR'd text
spell = SpellChecker()

# Tokenize the text into words
words = text.split()
corrected_words = []

for word in words:
    # Check if the word is misspelled
    if word not in spell:
        # Get the best correction
        correction = spell.correction(word)
        corrected_words.append(correction)
    else:
        corrected_words.append(word)

# Join the corrected words back into a single string
corrected_text = ' '.join(corrected_words)

# show the text after ocr-spellchecking has been applied
print("AFTER SPELLCHECK")
print("================")
print(corrected_text)
