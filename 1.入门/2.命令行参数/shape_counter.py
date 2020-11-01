import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input image")
ap.add_argument("-o", "--output", required=True, help="path to output image")
args = vars(ap.parse_args())

image = cv2.imread(args["input"])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)

cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in cnts:
    cv2.drawContours(image, [cnt], -1, (0, 0, 255), 2)

text = "I found {} total shapes".format(len(cnts))
cv2.putText(image, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)

cv2.imwrite(args["output"], image)