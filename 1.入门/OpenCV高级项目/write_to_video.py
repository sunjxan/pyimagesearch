import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2

from imutils.video_capture import playVideo

# 定义编解码器，fourcc是独立标示视频数据流格式的四字符代码
fourcc = cv2.VideoWriter_fourcc(*"XVID")
fps = 10.0
writer = None
output_filename = "vtest_output.avi"

def captureFunc(frame, frameIndex):
    global writer

    h, w = frame.shape[:2]

    if not writer:
        # 输出视频路径，编解码器，码率，分辨率
        writer = cv2.VideoWriter(output_filename, fourcc, fps, (w * 2, h * 2), True)

    zeros = np.zeros((h, w), dtype=np.uint8)
    blue = cv2.merge([frame[..., 0], zeros, zeros])
    green = cv2.merge([zeros, frame[..., 1], zeros])
    red = cv2.merge([zeros, zeros, frame[..., 2]])

    output = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    output[0:h, 0:w] = frame
    output[0:h, w:w*2] = red
    output[h:h*2, w:w*2] = green
    output[h:h * 2, 0:w] = blue

    # 写入文件
    writer.write(output)
    cv2.imshow("Output", output)

def endFunc():
    global writer
    if writer:
        writer.release()

playVideo("vtest.avi", fps=fps, captureFunc=captureFunc, showOriginalFrame=True, endFunc=endFunc)