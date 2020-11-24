import numpy as np

def non_max_suppression(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    pick = []

    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    # 计算边界框的面积并排序
    area = w * h
    idxs = np.argsort(area)

    # 循环直到idxs为空
    while len(idxs) > 0:
        # 先取最小面积的元素
        index = idxs[0]
        pick.append(index)
        idxs = np.delete(idxs, [0])

        # 计算其他所有元素与选中元素的重叠面积
        left = np.maximum(x[index], x[idxs])
        top = np.maximum(y[index], y[idxs])
        right = np.minimum(x[index] + w[index] - 1, x[idxs] + w[idxs] - 1)
        bottom = np.minimum(y[index] + h[index] - 1, y[idxs] + h[idxs] - 1)

        overlapW = np.maximum(0, right - left + 1)
        overlapH = np.maximum(0, bottom - top + 1)
        overlapArea = overlapW * overlapH

        # 计算重叠面积在两边元素面积分别的比例
        overlap1 = overlapArea / area[index]
        overlap2 = overlapArea / area[idxs]

        # 如果完全包含选中元素，或者重叠面积占自身面积比例超过overlapThresh，则被抑制
        idxs = np.delete(idxs, np.where(overlap1 == 1)[0])
        idxs = np.delete(idxs, np.where(overlap2 > overlapThresh)[0])

    return boxes[pick]