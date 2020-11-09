import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2

from imutils.video_capture import playVideo

# 模型是否初始化
modelInited = False
# 初始化模型需要帧数
initializationFrames = 120
# 已经提供给模型的帧数
frames = 0

# 创建背景减法器模型
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=initializationFrames)

def startFunc():
    print("[Info] start init model...")

def replayCondition(ret, frame):
    global modelInited
    global initializationFrames
    global frames

    # 如果模型刚好初始化完毕，重播视频，模型开始工作
    if not modelInited and frames == initializationFrames:
        print("[Info] model initialization complete, replay video")
        modelInited = True
        return True
    return False

def captureFunc(frame):
    global modelInited
    global frames

    # 为背景减法器提供帧
    mask = fgbg.apply(frame)
    frames += 1

    # 模型初始化结束
    if modelInited:
        # 形态学开运算
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=2)
        output = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("Mask", mask)
        cv2.imshow("Output", output)

def endFunc():
    global modelInited
    global initializationFrames
    global frames

    # 如果模型已经初始化，且已经开始工作，关闭该两个窗口
    if modelInited and frames > initializationFrames:
        cv2.destroyWindow("Mask")
        cv2.destroyWindow("Output")

playVideo("vtest.avi", fps=10.0, winname="Original", replayCondition=replayCondition, captureFunc=captureFunc, startFunc=startFunc, endFunc=endFunc)