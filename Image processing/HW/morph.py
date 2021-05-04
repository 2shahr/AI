import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import numpy as np
from skimage import color
import time

def im_erode(img, sl):
    p_img = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    p_img[1 : img.shape[0] + 1, 1 : img.shape[1] + 1] = img
    er_img = np.zeros_like(img)
    for i in range(1, img.shape[0] + 1):
        for j in range(1, img.shape[1] + 1):
            p = p_img[i - 1 : i + 2, j - 1 : j + 2]
            er_img[i - 1, j - 1] = (p * sl)[sl == 1].min()
    return er_img

def im_dilate(img, sl):
    p_img = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    p_img[1 : img.shape[0] + 1, 1 : img.shape[1] + 1] = img
    dl_img = np.zeros_like(img)
    for i in range(1, img.shape[0] + 1):
        for j in range(1, img.shape[1] + 1):
            p = p_img[i - 1 : i + 2, j - 1 : j + 2]
            dl_img[i - 1, j - 1] = (p * sl).max()
    return dl_img

def find_ccs(img):
    sl = np.ones((3, 3))
    cc = np.zeros(img.shape)

    i, j = 0, 0
    label = 1
    while True:
        c = np.zeros_like(img)
        flag = False
        while i < img.shape[0]:
            if j == img.shape[1]:
                j = 0
            while j < img.shape[1]:
                if img[i, j]:
                    flag = True
                    break
                j += 1
            if flag:
                break
            i += 1
        if not flag:
            break
        c[i, j] = 1
        s = 1
        while True:
            c = im_dilate(c, sl) & img
            temp = c.sum()
            if temp == s:
                break
            else:
                s = temp
            print(s)
        cc[c == 1] = label
        img[c == 1] = 0
        label += 1
    return cc
  
# Read Images 
img_1 = mpimg.imread('img1.tif', format='gray') 
img_1 = img_1[:, :, 1] > 0

sl_1 = np.ones((3, 3))
sl_2 = np.array([[0., 1, 0], [1, 1, 1], [0, 1, 0]])

img_1_dil_1 = im_dilate(img_1, sl_1)
img_1_dil_2 = im_dilate(img_1, sl_2)

img_1_er_1 = im_erode(img_1, sl_1)
img_1_er_2 = im_erode(img_1, sl_2)

img_2 = mpimg.imread('img2.tif', format='gray') 
img_2 = img_2[:, :, 1] > 0
img_2_dil = im_dilate(img_2, sl_1)
img_2_b = img_2_dil & np.logical_not(img_2)

img_3 = mpimg.imread('img3.tif', format='gray') 
img_3 = img_3 > 0

cc = find_ccs(img_3)
n_ccs = cc.max()
cc_colored = color.colorlabel.label2rgb(cc)

plt.figure()
plt.imshow(img_1_dil_1, cmap='gray')
plt.title('Dilated By SL 1')
plt.show(block=False)

plt.figure()
plt.imshow(img_1_dil_2, cmap='gray')
plt.title('Dilated By SL 2')
plt.show(block=False)

plt.figure()
plt.imshow(img_1_er_1, cmap='gray')
plt.title('Eroded By SL 1')
plt.show(block=False)

plt.figure()
plt.imshow(img_1_er_2, cmap='gray')
plt.title('Eroded By SL 2')
plt.show(block=False)

plt.figure()
plt.imshow(img_2_b, cmap='gray')
plt.show(block=False)

plt.figure()
plt.imshow(cc_colored)
plt.title('Number of CCs: {}'.format(n_ccs))
plt.show(block=True)



