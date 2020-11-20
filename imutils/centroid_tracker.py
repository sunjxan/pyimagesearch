import numpy as np
import cv2

import math
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        # 下一个对象ID
        self.nextObjectID = 0
        # 记录对象的质心
        self.objects = OrderedDict()
        # 记录对象最近连续消失的帧数
        self.disappeared = OrderedDict()
        # 保存对象需要的不连续消失帧数阈值
        self.maxDisappeared = maxDisappeared

    # 注册对象
    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    # 注销对象
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, centroids):
        # 如果数据为空，则对所有已注册对象，消失帧数加1，满足条件的注销
        updateCount = len(centroids)
        if updateCount == 0:
            # 将已注册对象字典转换为列表，因为迭代中操作字典，所以不要直接在字典上迭代
            objectKeys = list(self.objects.keys())
            for objectID in objectKeys:
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # 如果没有已注册的对象，则为所有传入质心注册对象
        objectCount = len(self.objects)
        if objectCount == 0:
            for centroid in centroids:
                self.register(centroid)
            return self.objects

        # 将已注册对象字典转换为列表
        objectKeys = list(self.objects.keys())
        objectValues = list(self.objects.values())
        # 为所有传入质心计算和已注册对象质心的距离
        D = np.ndarray((updateCount, objectCount), dtype=np.float32)
        for i in range(updateCount):
            for j in range(objectCount):
                D[i, j] = math.sqrt((centroids[i][0] - objectValues[j][0]) ** 2 + (centroids[i][1] - objectValues[j][1]) ** 2)

        # 每行距离中的最小值
        minValues = D.min(axis=1)
        # 每行距离最小值的下标
        colIndexs = D.argmin(axis=1)
        # 为每行距离最小值排序，返回下标列表
        sortedRowIndexs = minValues.argsort()
        # 按下标列表将每行距离最小值的下标重新排列
        sortedColIndexs = colIndexs[sortedRowIndexs]

        # 保存每个已注册对象是否已被匹配
        unused = [True] * objectCount
        # 为每个传入质心在已注册对象中进行匹配
        for row, col in zip(sortedRowIndexs, sortedColIndexs):
            if unused[col]:
                # 已注册对象为被匹配，则匹配成功
                unused[col] = False
                objectID = objectKeys[col]
                self.objects[objectID] = centroids[row]
                self.disappeared[objectID] = 0
            else:
                # 否则注册此质心
                self.register(centroids[row])

        # 对每个没出现的已注册对象，消失帧数加1，满足条件的注销
        for i, v in enumerate(unused):
            if v:
                objectID = objectKeys[i]
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

        return self.objects
