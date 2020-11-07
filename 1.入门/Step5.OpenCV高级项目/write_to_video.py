import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2

cap = cv2.VideoCapture()

if cap.open("vtest.avi"):
    # 定义编解码器，fourcc是独立标示视频数据流格式的四字符代码
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = None
    while True:
        ret, frame = cap.read()
        # 是否获得帧
        if not ret:
            break

        h, w = frame.shape[:2]
        fps = 10.0
        if not writer:
            # 输出视频路径，编解码器，码率，分辨率
            writer = cv2.VideoWriter("vtest_output.avi", None, fourcc, fps, (w * 2, h * 2), True)
            zeros = np.zeros((h, w), dtype=np.uint8)

        B, G, R = cv2.split(frame)
        blue = cv2.merge([B, zeros, zeros])
        green = cv2.merge([zeros, G, zeros])
        red = cv2.merge([zeros, zeros, R])

        output = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        output[0:h, 0:w] = frame
        output[0:h, w:w*2] = red
        output[h:h*2, w:w*2] = green
        output[h:h * 2, 0:w] = blue

        # 写入文件
        writer.write(output)
        cv2.imshow("Output", output)
        # 等待毫秒为 1000 / fps
        if cv2.waitKey(1000 // fps) & 0xFF == ord("q"):
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
