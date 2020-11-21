import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.switch_backend('GTK3Agg')

from imutils.video_capture import playVideo

# 模型是否初始化
modelInited = False
# 初始化模型需要帧数
initializationFrames = 120

# 创建背景减法器模型
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=initializationFrames)

def startFunc():
    print("[Info] start init model...")

def replayCondition(frame, frameIndex):
    global modelInited
    global initializationFrames

    # 如果模型刚好初始化完毕，重播视频，模型开始工作
    if not modelInited and frameIndex == initializationFrames:
        print("[Info] model initialization complete, replay video")
        modelInited = True
        return True
    return False

def captureFunc(frame, frameIndex):
    global modelInited

    # 为背景减法器提供帧
    mask = fgbg.apply(frame)

    # 模型初始化结束
    if modelInited:
        # 形态学开运算
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=2)
        output = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("Output", output)
        return frame

playVideo("vtest.avi", fps=10, replayCondition=replayCondition, captureFunc=captureFunc, startFunc=startFunc)