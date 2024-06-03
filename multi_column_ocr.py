from sklearn.cluster import AgglomerativeClustering
from pytesseract import Output
from tabulate import tabulate
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytesseract
import imutils
import cv2

# Define input image path and output CSV path directly
input_image_path = "Final OCR Code/ledger KANAK[1]_page-0001.jpg"  # Replace with your image path
output_csv_path = "Final OCR Code/results11.csv"  # Replace with your desired output path

# Set other parameters
min_conf = 0
dist_thresh = 25.0
min_size = 3

# set a seed for our random number generator
np.random.seed(42)

# load the input image and convert it to grayscale
image = cv2.imread(input_image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(20, 20))
plt.title('Gray Image')
plt.imshow(gray, cmap='gray')
plt.axis('off')
plt.show()

# initialize a rectangular kernel that is ~5x wider than it is tall,
# then smooth the image using a 3x3 Gaussian blur and then apply a
# blackhat morphological operator to find dark regions on a light
# background
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51,11))
gray = cv2.GaussianBlur(gray, (9,9), 0)
# gray = cv2.bilateralFilter(gray,9,75,75)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

# Display the Gaussian Blurred image
plt.figure(figsize=(20, 20))
plt.title('Gaussian Blurred Image')
plt.imshow(gray, cmap='gray')
plt.axis('off')
plt.show()


# Display the Blackhat image
plt.figure(figsize=(20, 20))
plt.title('Blackhat Image')
plt.imshow(blackhat, cmap='gray')
plt.axis('off')
plt.show()

# compute the Scharr gradient of the blackhat image and scale the
# result into the range [0, 255]
grad = cv2.Sobel(blackhat, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=-1)
grad = np.absolute(grad)
(minVal, maxVal) = (np.min(grad), np.max(grad))
grad = (grad - minVal) / (maxVal - minVal)
grad = (grad * 255).astype("uint8")

# Display the Scharr Gradient image
plt.figure(figsize=(20, 20))
plt.title('Scharr Gradient Image')
plt.imshow(grad, cmap='gray')
plt.axis('off')
plt.show()

# apply a closing operation using the rectangular kernel to close
# gaps in between characters, apply Otsu's thresholding method, and
# finally a dilation operation to enlarge foreground regions
grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, kernel)
thresh = cv2.threshold(grad, 0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)[1]
thresh = cv2.dilate(thresh, None, iterations=3)

# Display the final thresholded image
plt.figure(figsize=(20, 20))
plt.title('Final Thresholded Image')
plt.imshow(thresh, cmap='gray')
plt.axis('off')
plt.show()

# find contours in the thresholded image and grab the largest one,
# which we will assume is the table
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
tableCnt = max(cnts, key=cv2.contourArea)

# compute the bounding box coordinates of the stats table and extract
# the table from the input image
(x, y, w, h) = cv2.boundingRect(tableCnt)
table = image[y:y + h, x:x + w]

# Display the original input image
plt.figure(figsize=(20, 20))
plt.title('Original Input Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Display the extracted table image
plt.figure(figsize=(20, 20))
plt.title('Extracted Table Image')
plt.imshow(cv2.cvtColor(table, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# # Resize the extracted table image
# resized_table = cv2.resize(table, (0, 0), fx=2,fy=2)

# # Display the resized table image
# plt.figure(figsize=(20, 20))
# plt.title('Resized Table Image')
# plt.imshow(cv2.cvtColor(resized_table, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()

# set the PSM mode to detect sparse text, and then localize text in the table.
options = "--psm 11 --oem 3 -l eng --dpi 300 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.$%()-/"
results = pytesseract.image_to_data(cv2.cvtColor(table, cv2.COLOR_BGR2RGB), config=options, output_type=Output.DICT)


# print(results)

# df = pd.DataFrame(results)
# print(df)

# initialize a list to store the (x, y)-coordinates of the detected
# text along with the OCR'd text itself
coords = []
ocrText = []

# loop over each of the individual text localizations
for i in range(0, len(results["text"])):
    # extract the bounding box coordinates of the text region from
    # the current result
    x = results["left"][i]
    y = results["top"][i]
    w = results["width"][i]
    h = results["height"][i]
    # extract the OCR text itself along with the confidence of the
    # text localization
    text = results["text"][i]
    conf = int(results["conf"][i])

    # filter out weak confidence text localizations
    if conf > min_conf:
        # update our text bounding box coordinates and OCR'd text
        # respectively
        coords.append((x, y, w, h))
        ocrText.append(text)

# extract all x-coordinates from the text bounding boxes, setting the
# y-coordinate value to zero
xCoords = [(c[0], 0) for c in coords]

# apply hierarchical agglomerative clustering to the coordinates
clustering = AgglomerativeClustering(
    n_clusters=None,
    metric="euclidean",
    linkage="complete",
    distance_threshold=dist_thresh
)
clustering.fit(xCoords)

# initialize our list of sorted clusters
sortedClusters = []

# loop over all clusters
for l in np.unique(clustering.labels_):
    # extract the indexes for the coordinates belonging to the
    # current cluster
    idxs = np.where(clustering.labels_ == l)[0]

    # verify that the cluster is sufficiently large
    if len(idxs) > min_size:
        # compute the average x-coordinate value of the cluster and
        # update our clusters list with the current label and the
        # average x-coordinate
        avg = np.average([coords[i][0] for i in idxs])
        sortedClusters.append((l, avg))

# sort the clusters by their average x-coordinate and initialize our
# data frame
sortedClusters.sort(key=lambda x: x[1])
df = pd.DataFrame()

# loop over the clusters again, this time in sorted order
for (l, _) in sortedClusters:
    # extract the indexes for the coordinates belonging to the
    # current cluster
    idxs = np.where(clustering.labels_ == l)[0]
    # extract the y-coordinates from the elements in the current
    # cluster, then sort them from top-to-bottom
    yCoords = [coords[i][1] for i in idxs]
    sortedIdxs = idxs[np.argsort(yCoords)]

    # generate a random color for the cluster
    color = np.random.randint(0, 255, size=(3,), dtype="int")
    color = [int(c) for c in color]

    # loop over the sorted indexes
    for i in sortedIdxs:
        # extract the text bounding box coordinates and draw the
        # bounding box surrounding the current element
        (x, y, w, h) = coords[i]
        cv2.rectangle(table, (x, y), (x + w, y + h), color, 2)

    # extract the OCR'd text for the current column, then construct
    # a data frame for the data where the first entry in our column
    # serves as the header
    cols = [ocrText[i].strip() for i in sortedIdxs]
    currentDF = pd.DataFrame({cols[0]: cols[1:]})

    # concatenate *original* data frame with the *current* data
    # frame (we do this to handle columns that may have a varying
    # number of rows)
    df = pd.concat([df, currentDF], axis=1)

# Replace NaN values with an empty string and then show a nicely
# formatted version of your multi-column OCR'd text
df.fillna("", inplace=True)
print(tabulate(df, headers="keys", tablefmt="psql"))

# Write the table to disk as a CSV file
print("[INFO] saving CSV file to disk...")
df.to_csv(output_csv_path, index=False)

# Show the output image after performing multi-column OCR using matplotlib
plt.figure(figsize=(80, 80))
plt.title('Output Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
