import cv2
import numpy as np

class Stitcher:
    def stitch(self, images, ratio=.75, reprojThresh=4.0, showMatches=False):
        imagesCount = len(images)
        if imagesCount == 0:
            return
        if imagesCount == 1:
            return images[0]

        result = images[-1]
        for i in range(imagesCount - 1, 0, -1):
            result = self._stitch_two_images(imagesCount - i, images[i - 1], result, ratio, reprojThresh, showMatches)
            if result is None:
                return
        return result

    def _stitch_two_images(self, index, imageL, imageR, ratio, reprojThresh, showMatches):
        hL, wL = imageL.shape[:2]
        hR, wR = imageR.shape[:2]

        # SIFT获得关键点和特征向量
        kpsL, featuresL = self.detectAndDescribe(imageL)
        kpsR, featuresR = self.detectAndDescribe(imageR)

        # 匹配两个图像中的特征
        M = self.matchKeypoints(kpsL, kpsR, featuresL, featuresR, ratio, reprojThresh)
        if M is None:
            return

        matches, H, status = M
        # 透视变换，将右边图片变换
        result = cv2.warpPerspective(imageR, H, ((wL + wR), max(hL, hR)))
        # 再将左边图片覆盖在上层
        result[0:hL, 0:wL] = imageL

        # 获得拼接结果的外界矩形
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        cnts, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maxContour = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(maxContour)
        result = result[y:y+h, x:x+w]

        # 检查是否可视化关键点匹配
        if showMatches:
            vis = self.drawMatches(imageL, imageR, kpsL, kpsR, matches, status)
            cv2.imshow("Matches {}".format(index), vis)
        return result

    def detectAndDescribe(self, image):
        # 检测关键点并提取特征向量

        descriptor = cv2.xfeatures2d.SIFT_create()
        kps, features = descriptor.detectAndCompute(image, None)

        # 从关键点对象中获取坐标
        kps = np.array([kp.pt for kp in kps], dtype=np.float32)
        return kps, features

    def matchKeypoints(self, kpsL, kpsR, featuresL, featuresR, ratio, reprojThresh):

        # 特征匹配器，暴力穷举策略
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        # k近邻匹配，为featuresL中每个点在featuresR中寻找k个最近邻，结果列表中由近到远排列
        # 每个匹配项中queryIdx表示目标的featuresL下标，trainIdx表示目标的featuresR下标，distance表示两个关键点欧几里得距离
        rawMatches = matcher.knnMatch(featuresL, featuresR, 2)
        matches = []

        # 过滤假阳性匹配项
        for m in rawMatches:
            # Lowe's ratio test，检测有唯一最近邻
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].queryIdx, m[0].trainIdx))

        # 计算单应性至少需要4个匹配项
        if len(matches) > 4:
            # 构造两组点
            ptsL, ptsR = [], []
            for queryIdx, trainIdx in matches:
                ptsL.append(kpsL[queryIdx])
                ptsR.append(kpsR[trainIdx])
            ptsL = np.array(ptsL, dtype=np.float32)
            ptsR = np.array(ptsR, dtype=np.float32)

            # 计算两组点之间的单应性，返回变换矩阵H将关键点B投影到关键点A
            # 如果把左边图片按对应点变换到右边图片，结果图片展示不完全，
            # 所以应该将右边图片变换到左边图片
            H, status = cv2.findHomography(ptsR, ptsL, cv2.RANSAC, reprojThresh)

            return matches, H, status
        return

    def drawMatches(self, imageL, imageR, kpsL, kpsR, matches, status):
        # 绘制两个图像之间的关键点对应关系

        hL, wL = imageL.shape[:2]
        hR, wR = imageR.shape[:2]
        vis = np.zeros((max(hL, hR), wL + wR, 3), dtype=np.uint8)
        vis[0:hL, 0:wL] = imageL
        vis[0:hR, wL:] = imageR

        for ((queryIdx, trainIdx), s) in zip(matches, status):
            # 仅在关键点成功后处理匹配
            if s == 1:
                ptA = round(kpsL[queryIdx, 0].item()), round(kpsL[queryIdx, 1].item())
                ptB = round(kpsR[trainIdx, 0].item()) + wL, round(kpsR[trainIdx, 1].item())
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        return vis

    def removeBlackBorder(self, image, showAnimate=False, winname=None):
        def drawAnimate(mask, time):
            # 预览外接矩形内的拼接结果
            cv2.imshow(winname, cv2.bitwise_and(image, image, mask=mask))
            cv2.waitKey(time)

        # 四周加上10像素的黑色边框，保证可以从四个方向进行腐蚀
        image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
        # 得到目标区域轮廓
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        cnts, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maxContour = max(cnts, key=cv2.contourArea)
        # 得到目标区域外接矩形
        x, y, w, h = cv2.boundingRect(maxContour)
        boundingRect = np.zeros(thresh.shape, dtype=np.uint8)
        cv2.rectangle(boundingRect, (x, y), (x + w - 1, y + h - 1), 255, -1)
        if showAnimate:
            drawAnimate(boundingRect, 1000)

        # 外接矩形减去目标区域
        sub = cv2.subtract(boundingRect, thresh)
        # 腐蚀外接矩形，每次向内缩减1像素，直到完全在目标区域内部
        while cv2.countNonZero(sub) > 0:
            boundingRect = cv2.erode(boundingRect, None)
            if showAnimate:
                drawAnimate(boundingRect, 30)
            sub = cv2.subtract(boundingRect, thresh)

        # 得到新的矩形轮廓
        cnts, hier = cv2.findContours(boundingRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maxContour = max(cnts, key=cv2.contourArea)
        nX, nY, nW, nH = cv2.boundingRect(maxContour)

        left = nX
        right = nX + nW
        top = nY
        bottom = nY + nH

        # 1. 分别从左右两个方向对内部矩形进行膨胀，找到满足条件的最大矩形
        while left > x:
            left = left - 1
            boundingRect[top:bottom, left] = 255
            if showAnimate:
                drawAnimate(boundingRect, 30)
            sub = cv2.subtract(boundingRect, thresh)
            if cv2.countNonZero(sub) > 0:
                boundingRect[top:bottom, left] = 0
                if showAnimate:
                    drawAnimate(boundingRect, 30)
                left = left + 1
                break
        while right < x + w:
            right = right + 1
            boundingRect[top:bottom, right - 1] = 255
            if showAnimate:
                drawAnimate(boundingRect, 30)
            sub = cv2.subtract(boundingRect, thresh)
            if cv2.countNonZero(sub) > 0:
                boundingRect[top:bottom, right - 1] = 0
                if showAnimate:
                    drawAnimate(boundingRect, 30)
                right = right - 1
                break

        # 2. 分别从上下两个方向对内部矩形进行膨胀，找到满足条件的最大矩形
        while top > y:
            top = top - 1
            boundingRect[top, left:right] = 255
            if showAnimate:
                drawAnimate(boundingRect, 30)
            sub = cv2.subtract(boundingRect, thresh)
            if cv2.countNonZero(sub) > 0:
                boundingRect[top, left:right] = 0
                if showAnimate:
                    drawAnimate(boundingRect, 30)
                top = top + 1
                break
        while bottom < y + h:
            bottom = bottom + 1
            boundingRect[bottom - 1, left:right] = 255
            if showAnimate:
                drawAnimate(boundingRect, 30)
            sub = cv2.subtract(boundingRect, thresh)
            if cv2.countNonZero(sub) > 0:
                boundingRect[bottom - 1, left:right] = 0
                if showAnimate:
                    drawAnimate(boundingRect, 30)
                bottom = bottom - 1
                break

        # 获得没有黑色区域的拼接结果
        return image[top:bottom, left:right]