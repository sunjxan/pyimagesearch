import cv2

class ShapeDetector:
    def detect(self, cnt):
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, .04 * peri, True)
        count = len(approx)

        if count == 3:
            shape = "triangle"
        elif count == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ar = w / h
            shape = "square" if ar >= .95 and ar <= 1.05 else "rectangle"
        elif count == 5:
            shape = "pentagon"
        else:
            shape = "circle"
        return shape