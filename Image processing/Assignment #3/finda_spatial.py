from scipy import signal
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
#import numpy as np


text_img = cv2.imread('text.png', 0)
a_img = cv2.imread('a.png', 0)

h = a_img.shape[0]
w = a_img.shape[1]
c = signal.correlate2d(text_img, a_img)

m = c.max()
inds = (c == m).nonzero()
plt.figure()
ax = plt.subplots(1)
ax[1].imshow(text_img, cmap='gray')
for i in range(len(inds[0])):
    rec = Rectangle((inds[1][i] - w, inds[0][i] - h), w, h, edgecolor='r', facecolor='none')
    ax[1].add_patch(rec)

a_img = cv2.rotate(a_img, 2)
h = a_img.shape[0]
w = a_img.shape[1]
c = signal.correlate2d(text_img, a_img)

m = c.max()
inds = (c == m).nonzero()
ax[1].imshow(text_img, cmap='gray')
for i in range(len(inds[0])):
    rec = Rectangle((inds[1][i] - w, inds[0][i] - h), w, h, edgecolor='r', facecolor='none')
    ax[1].add_patch(rec)
    
plt.show()





