import os

import numpy as np
import cv2

KEYCODE_PAUSE = ord(" ")
KEYCODE_RESUME = ord(" ")
KEYCODE_QUIT = ord("q")
KEYCODE_REPLAY = ord("r")

def captureCamera(deviceIndex=0, fps=1.0, winname="", quitCondition=None, captureFunc=None, startFunc=None, endFunc=None):
    # 处理参数
    if fps <= 0:
        fps = 1.0
    if not winname:
        winname = "Camera {}".format(deviceIndex)

    # 创建视频捕捉器
    cap = cv2.VideoCapture()
    # 打开摄像头
    cap.open(deviceIndex)

    if cap.isOpened() and startFunc:
        startFunc()

    # 如果捕捉器是打开的
    if cap.isOpened():
        # 保存键盘输入
        keyCode = None

        while True:
            # 获取帧
            ret, frame = cap.read()
            # 是否获得帧
            if not ret:
                break

            # 判断是否退出
            if quitCondition and quitCondition(frame):
                keyCode = KEYCODE_QUIT
                break

            cv2.imshow(winname, frame)
            if captureFunc:
                captureFunc(frame.copy())
            # 等待时间为 1000 / fps
            keyCode = cv2.waitKey(round(1000 / fps))

            # 暂停
            if keyCode == KEYCODE_PAUSE:
                while True:
                    keyCode = cv2.waitKey()
                    if keyCode == KEYCODE_RESUME or keyCode == KEYCODE_QUIT:
                        break
            # 退出
            if keyCode == KEYCODE_QUIT:
                break

        # 释放捕捉器
        cap.release()
        cv2.destroyWindow(winname)
        if endFunc:
            endFunc()
        return True
    else:
        print("[Info] can't open camera")
        return False

def playVideo(filename, fps=10.0, winname="", quitCondition=None, replayCondition=None, captureFunc=None, startFunc=None, endFunc=None):
    # 处理参数
    if fps <= 0:
        fps = 10.0
    if not winname:
        pos = filename.rfind(os.sep)
        if pos == -1:
            winname = filename
        else:
            winname = filename[pos+1:]

    # 创建视频捕捉器
    cap = cv2.VideoCapture()
    # 打开视频文件
    cap.open(filename)

    if cap.isOpened() and startFunc:
        startFunc()

    # 如果捕捉器是打开的
    while cap.isOpened():
        # 保存键盘输入
        keyCode = None

        while True:
            # 获取帧
            ret, frame = cap.read()

            # 判断是否重播
            if replayCondition and replayCondition(ret, frame):
                keyCode = KEYCODE_REPLAY
                break

            # 是否获得帧
            if not ret:
                break

            # 判断是否退出
            if quitCondition and quitCondition(frame):
                keyCode = KEYCODE_QUIT
                break

            cv2.imshow(winname, frame)
            if captureFunc:
                captureFunc(frame.copy())
            # 等待时间为 1000 / fps
            keyCode = cv2.waitKey(round(1000 / fps))

            # 暂停
            if keyCode == KEYCODE_PAUSE:
                while True:
                    keyCode = cv2.waitKey()
                    if keyCode == KEYCODE_RESUME or keyCode == KEYCODE_QUIT or keyCode == KEYCODE_REPLAY:
                        break
            # 退出或重播
            if keyCode == KEYCODE_QUIT or keyCode == KEYCODE_REPLAY:
                break

        # 释放捕捉器
        cap.release()

        # 重播
        if keyCode == KEYCODE_REPLAY:
            cap.open(filename)
            continue

        # 退出
        cv2.destroyWindow(winname)
        if endFunc:
            endFunc()
        break
    else:
        print("[Info] can't open file")
        return False
    return True