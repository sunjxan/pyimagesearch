import cv2
import numpy as np

class Stitcher:
    def stitch(self, images, ratio=.75, reprojThresh=4.0, showMatches=False):
        imageA, imageB = images
        hA, wA = imageA.shape[:2]
        hB, wB = imageB.shape[:2]

        # SIFT获得关键点和特征向量
        kpsA, featuresA = self.detectAndDescribe(imageA)
        kpsB, featuresB = self.detectAndDescribe(imageB)

        # 匹配两个图像中的特征
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        if not M:
            return

        matches, H, status = M
        # 透视变换，将右边图片变换
        result = cv2.warpPerspective(imageB, H, ((wA + wB), hA))
        # 再将左边图片覆盖在上层
        result[0:hA, 0:wA] = imageA

        # 检查是否可视化关键点匹配
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            return result, vis
        return result

    def detectAndDescribe(self, image):
        # 检测关键点并提取特征向量

        descriptor = cv2.xfeatures2d.SIFT_create()
        kps, features = descriptor.detectAndCompute(image, None)

        # 从关键点对象中获取坐标
        kps = np.array([kp.pt for kp in kps], dtype=np.float32)
        return kps, features

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):

        # 特征匹配器，暴力穷举策略
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        # k近邻匹配，为featuresA中每个点在featuresB中寻找k个最近邻，结果列表中由近到远排列
        # 每个匹配项中queryIdx表示目标的featuresA下标，trainIdx表示目标的featuresB下标，distance表示两个关键点欧几里得距离
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # 过滤假阳性匹配项
        for m in rawMatches:
            # Lowe's ratio test，检测有唯一最近邻
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].queryIdx, m[0].trainIdx))

        # 计算单应性至少需要4个匹配项
        if len(matches) > 4:
            # 构造两组点
            ptsA, ptsB = [], []
            for queryIdx, trainIdx in matches:
                ptsA.append(kpsA[queryIdx])
                ptsB.append(kpsB[trainIdx])
            ptsA = np.array(ptsA, dtype=np.float32)
            ptsB = np.array(ptsB, dtype=np.float32)

            # 计算两组点之间的单应性，返回变换矩阵H将关键点B投影到关键点A
            # 如果把左边图片按对应点变换到右边图片，结果图片展示不完全，
            # 所以应该将右边图片变换到左边图片
            H, status = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, reprojThresh)

            return matches, H, status
        return

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 绘制两个图像之间的关键点对应关系

        hA, wA = imageA.shape[:2]
        hB, wB = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype=np.uint8)
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        for ((queryIdx, trainIdx), s) in zip(matches, status):
            # 仅在关键点成功后处理匹配
            if s == 1:
                ptA = round(kpsA[queryIdx, 0].item()), round(kpsA[queryIdx, 1].item())
                ptB = round(kpsB[trainIdx, 0].item()) + wA, round(kpsB[trainIdx, 1].item())
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        return vis