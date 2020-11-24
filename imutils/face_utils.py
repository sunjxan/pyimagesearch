from collections import OrderedDict

import numpy as np
import cv2

FACIAL_LANDMARKS_IDXS = OrderedDict([
	("jaw", (0, 17)),				# 颚线
	("right_eyebrow", (17, 22)),	# 右眉
	("left_eyebrow", (22, 27)),		# 左眉
	("nose", (27, 36)),				# 鼻子
	("right_eye", (36, 42)),		# 右眼
	("left_eye", (42, 48)),			# 左眼
	("mouth", (48, 68))				# 嘴巴
])

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x + 1
	h = rect.bottom() - y + 1
	return (x, y, w, h)

def shape_to_np(shape):
    return np.array(list(map(lambda pt: (pt.x, pt.y), shape.parts())), dtype=np.int32)