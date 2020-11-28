import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append("..")

import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.switch_backend('GTK3Agg')

import math
import face_recognition

import imutils

def getMaxContour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    cnts, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return
    maxContour = max(cnts, key=cv2.contourArea)
    return maxContour

def overlay_image(bg, fg, bgPoint=(0, 0), fgPoint=(0, 0)):
    # 获得最大轮廓
    gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 192, 255, cv2.THRESH_BINARY_INV)
    cnts, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return bg
    maxContour = max(cnts, key=cv2.contourArea)

    # 根据最大轮廓获得由0和1组成的权值矩阵
    alpha = np.zeros(fg.shape[:2], dtype=np.uint8)
    cv2.drawContours(alpha, [maxContour], -1, 1, -1)
    alpha = cv2.merge([alpha, alpha, alpha])

    # 解决超出背景范围的问题，获得ROI
    start = np.array(bgPoint) - np.array(fgPoint)
    end = np.array((start[0] + fg.shape[1], start[1] + fg.shape[0]))
    shape = bg[start[1]:end[1], start[0]:end[0]].shape[:2]
    roi = bg[start[1]:start[1]+shape[0], start[0]:start[0]+shape[1]]

    # 根据ROI调整前景范围
    cut = np.zeros_like(start)
    if start[0] < 0:
        cut[0] = -start[0]
        start[0] = 0
    if start[1] < 0:
        cut[1] = -start[1]
        start[1] = 0
    fg = fg[cut[1]:cut[1]+shape[0], cut[0]:cut[0]+shape[1]]
    alpha = alpha[cut[1]:cut[1]+shape[0], cut[0]:cut[0]+shape[1]]

    # 混合图片，并赋值
    output = cv2.add(cv2.multiply(fg, alpha), cv2.multiply(roi, 1-alpha))
    bg[start[1]:start[1]+shape[0], start[0]:start[0]+shape[1]] = output

    return bg

def create_gif(image, faces):
    if len(faces) == 0:
        return

    # 中间步数
    steps = 20
    output = [None] * steps
    for face in faces:
        # 生成中间步数的坐标列表
        sgYs = np.linspace(0, face['bgCenterSG'][1], steps).round().astype(np.int)
        cigarYs = np.linspace(image.shape[0], face['bgCenterCigar'][1], steps).round().astype(np.int)
        for i in range(steps):
            if output[i] is None:
                img = image.copy()
            else:
                img = output[i]
            img = overlay_image(img, face['sg'], bgPoint=(face['bgCenterSG'][0], sgYs[i]), fgPoint=face['fgCenterSG'])
            img = overlay_image(img, face['cigar'], bgPoint=(face['bgCenterCigar'][0], cigarYs[i]), fgPoint=face['fgCenterCigar'])
            output[i] = img
    # 创建目录，保存中间图片
    os.system('rm -rf ./create_gif')
    os.mkdir('./create_gif')
    imagePaths = []
    for i in range(steps):
        imagePath = './create_gif/{}.png'.format(str(i).zfill(3))
        cv2.imwrite(imagePath, output[i])
        imagePaths.append(imagePath)
    lastPath = imagePaths[-1]
    imagePaths = imagePaths[:-1]

    # 准备参数
    delay = 10
    finalDelay = 120
    loop = 0
    # 安装imagemagick后，使用convert命令生成gif
    cmd = "convert -delay {} {} -delay {} {} -loop {} {}".format(delay, " ".join(imagePaths), finalDelay, lastPath, loop, 'create_gif.gif')
    os.system(cmd)

    # 删除中间目录
    os.system('rm -rf ./create_gif')

image_sg = cv2.imread('sunglasses.png')
image_cigar = cv2.imread('cigar.png')

image = cv2.imread('faces.jpg')
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

boxes = face_recognition.face_locations(rgb, model='cnn')
landmarks = face_recognition.face_landmarks(rgb, boxes)

faces = []
for marks in landmarks:
    # 左右眼轮廓
    leftEyePts = marks['left_eye']
    rightEyePts = marks['right_eye']
    # 左右眼中心
    leftEyeCenter = np.array(leftEyePts).mean(axis=0).round().astype(np.int)
    rightEyeCenter = np.array(rightEyePts).mean(axis=0).round().astype(np.int)

    # 计算眼镜旋转角度
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = -np.degrees(np.arctan2(dY, dX))
    # 计算眼镜宽度
    dis = math.sqrt(dY ** 2 + dX ** 2)
    width = dis * 2
    # 调整眼镜
    sg = imutils.resize(image_sg, width=width)
    maxContour = getMaxContour(sg)
    sg = imutils.rotate_contour(sg, maxContour, angle, bgColor=(255, 255, 255))

    # 眼镜位置
    bgCenterSG = ((leftEyeCenter + rightEyeCenter) / 2).round().astype(np.int)
    fgCenterSG = (round(sg.shape[1] / 2), round(sg.shape[0] / 2))

    # 上下嘴唇轮廓
    topLipPts = marks['top_lip']
    bottomLipPts = marks['bottom_lip']
    # 上下嘴唇中心
    topLipCenter = np.array(topLipPts).mean(axis=0).round().astype(np.int)
    bottomLipCenter = np.array(bottomLipPts).mean(axis=0).round().astype(np.int)
    # 调整雪茄
    cigar = imutils.resize(image_cigar, width=width * .55)
    maxContour = getMaxContour(cigar)
    cigar = imutils.rotate_contour(cigar, maxContour, angle, bgColor=(255, 255, 255))

    # 雪茄位置
    bgCenterCigar = ((topLipCenter + bottomLipCenter) / 2).round().astype(np.int)
    fgCenterCigar = (0, 0)

    # 准备数据
    faces.append({
        'sg': sg,
        'bgCenterSG': bgCenterSG,
        'fgCenterSG': fgCenterSG,
        'cigar': cigar,
        'bgCenterCigar': bgCenterCigar,
        'fgCenterCigar': fgCenterCigar
    })

# 制作gif
create_gif(image, faces)