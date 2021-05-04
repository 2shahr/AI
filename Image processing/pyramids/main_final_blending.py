from cv2 import cv2 as cv
import numpy as np
import copy
import matplotlib.pyplot as plt


apple = cv.imread('apple.png')
apple = cv.resize(apple, (256, 256))

orange = cv.imread('orange.png')
orange = cv.resize(orange, (256, 256))

apple_gp = [apple]
orange_gp = [orange]

n_layers = 5

for i in range(n_layers):
    apple = cv.pyrDown(apple)
    apple_gp.append(apple)

    orange = cv.pyrDown(orange)
    orange_gp.append(orange)
    

apple_lp = [apple_gp[n_layers]]
orange_lp = [orange_gp[n_layers]]
for i in range(n_layers, 0, -1):
    up = cv.pyrUp(apple_gp[i])
    sub = cv.subtract(apple_gp[i - 1], up)
    apple_lp.append(sub)

    up = cv.pyrUp(orange_gp[i])
    sub = cv.subtract(orange_gp[i - 1], up)
    orange_lp.append(sub)

merge = []
for la, lo in zip(apple_lp, orange_lp):
    m, n, _ = la.shape

    mask = np.zeros((m, n))
    mask[np.tril_indices(m, -1)] = 1
    for i in range(0, m, 2):
        mask[i, i] = 1
    mask_a = np.stack((mask, mask, mask), axis=2)

    mask = np.zeros((m, n))
    mask[np.triu_indices(m, 0)] = 1
    for i in range(0, m, 2):
        mask[i, i] = 0
    mask_o = np.stack((mask, mask, mask), axis=2)

    a = np.multiply(la, mask_a)
    o = np.multiply(lo, mask_o)

    m = np.add(a, o)
    merge.append(m)
    
apple_orange = merge[0]
for i in range(1, n_layers + 1):
    apple_orange = cv.pyrUp(apple_orange)
    apple_orange = cv.add(apple_orange, merge[i])

cv.imwrite('apple_orange.jpg', apple_orange)

