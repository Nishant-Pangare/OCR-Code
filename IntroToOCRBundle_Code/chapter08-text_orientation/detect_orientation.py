# USAGE
# python detect_orientation.py --image images/rotated_90_clockwise.png

# import the necessary packages
from pytesseract import Output
import pytesseract
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image to be OCR'd")
args = vars(ap.parse_args())

# load the input image, convert it from BGR to RGB channel ordering,
# and use Tesseract to determine the text orientation
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)

# display the orientation information
print("[INFO] detected orientation: {}".format(
    results["orientation"]))
print("[INFO] rotate by {} degrees to correct".format(
    results["rotate"]))
print("[INFO] detected script: {}".format(results["script"]))

# rotate the image to correct the orientation
rotated = imutils.rotate_bound(image, angle=results["rotate"])

# show the original image and output image after orientation correction using matplotlib
plt.figure(figsize=(10, 5))

# Display original image
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Display rotated image
plt.subplot(1, 2, 2)
plt.title("Output")
plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Show both images
plt.show()
