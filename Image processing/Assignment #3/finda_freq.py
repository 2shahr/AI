from scipy import signal
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def process_in_freq_domain(text_img, a_img):
    a_img = a_img[::-1, ::-1]
    max_w = max(text_img.shape[1], a_img.shape[1])
    max_h = max(text_img.shape[0], a_img.shape[0])
    
    text_img = np.pad(text_img, 
                      [(0, max_h - text_img.shape[0]), 
                       (0, max_w - text_img.shape[1])], 
                      mode='constant', constant_values=0)
    a_img = np.pad(a_img, 
                      [(0, max_h - a_img.shape[0]), 
                       (0, max_w - a_img.shape[1])], 
                      mode='constant', constant_values=0)

    text_img_fft = np.fft.fft2(text_img)
    a_img_fft = np.fft.fft2(a_img)
    
    return np.real(np.fft.ifft2(text_img_fft * a_img_fft))

text_img = cv2.imread('text.png', 0)
a_img = cv2.imread('a.png', 0)

h = a_img.shape[0]
w = a_img.shape[1]
c = process_in_freq_domain(text_img, a_img)

m = c.max()
inds = (c >= 0.99 * m).nonzero()
plt.figure()
ax = plt.subplots(1)
for i in range(len(inds[0])):
    rec = Rectangle((inds[1][i] - w, inds[0][i] - h), w, h, edgecolor='r', facecolor='none')
    ax[1].add_patch(rec)

a_img = cv2.rotate(a_img, 2)
h = a_img.shape[0]
w = a_img.shape[1]
c = process_in_freq_domain(text_img, a_img)

m = c.max()
inds = (c >= 0.99 * m).nonzero()
ax[1].imshow(text_img, cmap='gray')
for i in range(len(inds[0])):
    rec = Rectangle((inds[1][i] - w, inds[0][i] - h), w, h, edgecolor='r', facecolor='none')
    ax[1].add_patch(rec)
    
plt.show()

