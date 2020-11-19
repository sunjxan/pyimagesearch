import os
import datetime

import numpy as np
import cv2

KEYCODE_PAUSE = ord(" ")
KEYCODE_RESUME = ord(" ")
KEYCODE_QUIT = ord("q")
KEYCODE_REPLAY = ord("r")
KEYCODE_SCREENSHOT = ord("s")

def captureCamera(deviceIndex=0, fps=1000.0, winname="", quitCondition=None, captureFunc=None, startFunc=None, endFunc=None):
    # 处理参数
    if fps < 1e-5 or fps > 1e3:
        fps = 1000.0
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

        # 帧序号
        frameIndex = 0
        while True:
            # 获取帧
            ret, frame = cap.read()
            # 是否获得帧
            if not ret:
                break

            # 判断是否退出
            if quitCondition and quitCondition(frame, frameIndex):
                keyCode = KEYCODE_QUIT
                break

            cv2.imshow(winname, frame)
            if captureFunc:
                captureFunc(frame.copy(), frameIndex)
            # 等待时间为 1000 / fps
            keyCode = cv2.waitKey(round(1000 / fps))

            # 截图
            if keyCode == KEYCODE_SCREENSHOT:
                now = datetime.datetime.now()
                cv2.imwrite('camera_{}_{}_{}.png'.format(deviceIndex, now.date(), str(now.time()).replace(':', '-')), frame)
                
            # 暂停
            if keyCode == KEYCODE_PAUSE:
                while True:
                    keyCode = cv2.waitKey()
                    if keyCode == KEYCODE_RESUME or keyCode == KEYCODE_QUIT:
                        break
                    # 截图
                    if keyCode == KEYCODE_SCREENSHOT:
                        now = datetime.datetime.now()
                        cv2.imwrite(
                            'camera_{}_{}_{}.png'.format(deviceIndex, now.date(), str(now.time()).replace(':', '-')),
                            frame)
            # 退出
            if keyCode == KEYCODE_QUIT:
                break

            frameIndex += 1

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
    if fps < 1e-5 or fps > 1e3:
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

        # 帧序号
        frameIndex = 0
        while True:
            # 获取帧
            ret, frame = cap.read()

            # 是否获得帧
            if not ret:
                break

            # 判断是否退出
            if quitCondition and quitCondition(frame, frameIndex):
                keyCode = KEYCODE_QUIT
                break

            # 判断是否重播
            if replayCondition and replayCondition(frame, frameIndex):
                keyCode = KEYCODE_REPLAY
                break

            cv2.imshow(winname, frame)
            if captureFunc:
                captureFunc(frame.copy(), frameIndex)
            # 等待时间为 1000 / fps
            keyCode = cv2.waitKey(round(1000 / fps))

            # 截图
            if keyCode == KEYCODE_SCREENSHOT:
                now = datetime.datetime.now()
                cv2.imwrite('{}_{}_{}_{}.png'.format(filename, frameIndex, now.date(), str(now.time()).replace(':', '-')), frame)

            # 暂停
            if keyCode == KEYCODE_PAUSE:
                while True:
                    keyCode = cv2.waitKey()
                    if keyCode == KEYCODE_RESUME or keyCode == KEYCODE_QUIT or keyCode == KEYCODE_REPLAY:
                        break
                    # 截图
                    if keyCode == KEYCODE_SCREENSHOT:
                        now = datetime.datetime.now()
                        cv2.imwrite('{}_{}_{}_{}.png'.format(filename, frameIndex, now.date(),
                                                             str(now.time()).replace(':', '-')), frame)
            # 退出或重播
            if keyCode == KEYCODE_QUIT or keyCode == KEYCODE_REPLAY:
                break
            
            frameIndex += 1

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