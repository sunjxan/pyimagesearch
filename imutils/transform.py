import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)

    # 和最小是左上，和最大是右下
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 差最小是右上，差最大是左下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # [左上， 右上， 右下， 左下]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect

    widthA = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
    widthB = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
    maxWidth = np.round(max(widthA, widthB)).astype(np.int32)

    heightA = np.sqrt((bl[0] - tl[0]) ** 2 + (bl[1] - tl[1]) ** 2)
    heightB = np.sqrt((br[0] - tr[0]) ** 2 + (br[1] - tr[1]) ** 2)
    maxHeight = np.round(max(heightA, heightB)).astype(np.int32)

    dst = np.array([
        (0, 0),
        (maxWidth - 1, 0),
        (maxWidth - 1, maxHeight - 1),
        (0, maxHeight - 1)
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped