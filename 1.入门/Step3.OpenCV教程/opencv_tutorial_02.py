import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2

image = cv2.imread("tetris_blocks.png")
cv2.imshow("Image", image)
cv2.waitKey()
cv2.destroyAllWindows()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey()
cv2.destroyAllWindows()

edged = cv2.Canny(gray, 30, 150)
cv2.imshow("Edged", edged)
cv2.waitKey()
cv2.destroyAllWindows()

ret, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Thresh", thresh)
cv2.waitKey()
cv2.destroyAllWindows()

cnts, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = image.copy()

for cnt in cnts:
    cv2.drawContours(output, [cnt], -1, (240, 0, 159), 3)
    cv2.imshow("Contours", output)
    cv2.waitKey(300)

text = "I found {} objects!".format(len(cnts))
cv2.putText(output, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (240, 0, 159), 2)
cv2.imshow("Contours", output)
cv2.waitKey()
cv2.destroyAllWindows()

mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=5)
cv2.imshow("Eroded", mask)
cv2.waitKey()
cv2.destroyAllWindows()

mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=2)
cv2.imshow("Dilated", mask)
cv2.waitKey()
cv2.destroyAllWindows()

mask = thresh.copy()
output = cv2.bitwise_not(image, mask=mask)
cv2.imshow("Not Image", output)
cv2.waitKey()
cv2.destroyAllWindows()

mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Image And Image", output)
cv2.waitKey()
cv2.destroyAllWindows()

mask = thresh.copy()
output = cv2.bitwise_or(image, image, mask=mask)
cv2.imshow("Image Or Image", output)
cv2.waitKey()
cv2.destroyAllWindows()

mask = thresh.copy()
output = cv2.bitwise_xor(image, image, mask=mask)
cv2.imshow("Image Xor Image", output)
cv2.waitKey()
cv2.destroyAllWindows()
