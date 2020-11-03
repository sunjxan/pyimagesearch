import cv2
import numpy as np
import math

class ColorLabeler:
    def label(self, image, cnt):
        colors = ["red", "yellow", "green", "cyan", "blue", "magenta"]

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        hue = cv2.mean(hls[:, :, 0], mask=mask)[0]
        hue =  hue + 15 if hue < 165 else hue - 165
        return colors[math.floor(hue / 30)]
