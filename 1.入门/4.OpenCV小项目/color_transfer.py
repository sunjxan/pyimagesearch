import os, sys
os.environ["DISPLAY"] = "windows:0"
sys.path.append('../..')

import cv2
import numpy as np

def color_transfer(sourcePath, targetPath):
    source = cv2.imread(sourcePath)
    L, A, B = cv2.split(cv2.cvtColor(source, cv2.COLOR_BGR2LAB))
    L_mean, L_std = L.mean(), L.std()
    A_mean, A_std = A.mean(), A.std()
    B_mean, B_std = B.mean(), B.std()
    target = cv2.imread(targetPath)
    l, a, b = cv2.split(cv2.cvtColor(target, cv2.COLOR_BGR2LAB))
    l = (l - l.mean()) / l.std() * L_std + L_mean
    a = (a - a.mean()) / a.std() * A_std + A_mean
    b = (b - b.mean()) / b.std() * B_std + B_mean
    result = cv2.merge([l, a, b]).round().clip(0, 255).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    cv2.imshow("Source", source)
    cv2.imshow("Target", target)
    cv2.imshow("Result", result)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return result

color_transfer("ocean_sunset.jpg", "ocean_day.jpg")

color_transfer("ocean_day.jpg", "ocean_sunset.jpg")

color_transfer('autumn.jpg', 'fallingwater.jpg')

color_transfer('fallingwater.jpg', 'autumn.jpg')

color_transfer('woods.jpg', 'storm.jpg')

color_transfer('storm.jpg', 'woods.jpg')