# 显著性检测，就是使用图像处理技术和计算机视觉算法来定位图片中最“显著”的区域。
# 显著区域就是指图片中引人注目的区域或比较重要的区域，例如人眼在观看一幅图片时会首先关注的区域。
# 显著性检测在目标检测、机器人领域有很多应用。

import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("../..")

import numpy as np
import cv2

from imutils.video_capture import playVideo

# 检测器
saliency = None

def StaticSaliencySpectralResidual_startFunc():
    global saliency
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

def StaticSaliencySpectralResidual_captureFunc(frame):
    global saliency
    ret, saliencyMap = saliency.computeSaliency(frame)
    saliencyMap = (saliencyMap * 255).round().astype(np.uint8)
    cv2.imshow("Output", saliencyMap)

def StaticSaliencyFineGrained_startFunc():
    global saliency
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()

def StaticSaliencyFineGrained_captureFunc(frame):
    global saliency
    ret, saliencyMap = saliency.computeSaliency(frame)
    saliencyMap = (saliencyMap * 255).round().astype(np.uint8)
    cv2.imshow("Output", saliencyMap)

motionSaliencyInited = False
motionSaliencyComplete = False

def MotionSaliencyBinWangApr2014_startFunc():
    global saliency
    saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()

def MotionSaliencyBinWangApr2014_replayCondition(ret, frame):
    global motionSaliencyComplete

    # 如果模型刚好初始化完毕，重播视频，模型开始工作
    if motionSaliencyInited and not motionSaliencyComplete:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, saliencyMap = saliency.computeSaliency(frame)
        if (saliencyMap != 1).any():
            motionSaliencyComplete = True
            return True
    return False

def MotionSaliencyBinWangApr2014_captureFunc(frame):
    global motionSaliencyInited
    global motionSaliencyComplete
    global saliency

    if not motionSaliencyInited:
        motionSaliencyInited = True
        saliency.setImagesize(frame.shape[1], frame.shape[0])
        saliency.init()

    if motionSaliencyComplete:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, saliencyMap = saliency.computeSaliency(frame)
        saliencyMap = (saliencyMap * 255).round().astype(np.uint8)
        cv2.imshow("Output", saliencyMap)

def endFunc():
    cv2.destroyAllWindows()

# 静态显着性检测（此类显着性检测算法依赖于图像特征和统计信息来定位图像中显著性区域）

# 静态频谱显着性
playVideo("vtest.avi", fps=10.0, winname="Input", startFunc=StaticSaliencySpectralResidual_startFunc, captureFunc=StaticSaliencySpectralResidual_captureFunc, endFunc=endFunc)

# 细粒度显着性
playVideo("vtest.avi", fps=10.0, winname="Input", startFunc=StaticSaliencyFineGrained_startFunc, captureFunc=StaticSaliencyFineGrained_captureFunc, endFunc=endFunc)

# 运动显着性检测（此类显着性检测算法输入为视频或一系列连续帧。运动显着性算法处理这些连续的帧，并跟踪帧中“移动”的对象。这些移动的对象被认为是显着性区域）
playVideo("vtest.avi", fps=10.0, winname="Input", replayCondition=MotionSaliencyBinWangApr2014_replayCondition, startFunc=MotionSaliencyBinWangApr2014_startFunc, captureFunc=MotionSaliencyBinWangApr2014_captureFunc, endFunc=endFunc)