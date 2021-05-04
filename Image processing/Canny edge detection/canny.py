import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from queue import Queue
from skimage import feature

def gaussian(size, sigma):
    k = int(size) // 2
    [x, y] = np.mgrid[-k : k + 1, -k : k + 1]
    mask = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return mask

def sobel():
    s_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    s_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    return s_x, s_y

def apply_guassian(image, size, sigma):
    g = gaussian(size, sigma)
    c_image = convolve(image, g)
    return c_image

def convolve(image, kernel):
    h = kernel.shape[0] // 2
    w = kernel.shape[1] // 2
    
    img = np.zeros((image.shape[0] + kernel.shape[0], image.shape[1] + kernel.shape[1]))
    img[h : h + image.shape[0], w : w + image.shape[1]] = image
    conv_img = np.zeros(img.shape)
    
    for i in range(h, img.shape[0] - h):
        for j in range(w, img.shape[1] - w):
            conv_img[i, j] = (kernel * img[i - h : i + h + 1, j - w : j + w + 1]).sum()
    
    return conv_img[h : h + image.shape[0], w : w + image.shape[1]]
    
def apply_sobel(image):
    s_x, s_y = sobel()
    i_x = convolve(image, s_x)
    i_y = convolve(image, s_y)
    g = np.sqrt(i_x ** 2 + i_y ** 2)
    theta = np.arctan2(i_y, i_x)
    return g, theta

def non_maximum_suppresion(g, theta):
    mask = np.zeros(g.shape, dtype = np.float32)
    pi = np.pi
    for i in range(1, g.shape[0] - 1):
        for j in range(1, g.shape[1] - 1):
            if (-pi / 8  <= theta[i, j] < pi / 8) or             (np.pi - pi / 8 <= theta[i, j] < pi) or (-pi <= theta[i, j] < -pi + pi / 8):
                pre_g = g[i, j - 1]
                nxt_g = g[i, j + 1]
                
            elif (pi / 8 <= theta[i, j] < 3 * pi / 8) or (-pi + pi / 8 <= theta[i, j] < -pi + 3 * pi / 8):
                pre_g = g[i - 1, j + 1]
                nxt_g = g[i + 1, j - 1]
                
            elif (3 * pi / 8 <= theta[i, j] <= 5 * pi / 8) or (-pi + 3 * pi / 8 <= theta[i, j] <= -pi - 5 * pi / 8):
                pre_g = g[i + 1, j]
                nxt_g = g[i - 1, j]
            else:
                pre_g = g[i - 1, j - 1]
                nxt_g = g[i + 1, j + 1]
            
            if g[i, j] >= pre_g and g[i, j] >= nxt_g:
                mask[i, j] = g[i, j]
    return mask

def threshold(image, low_th, high_th):
    mask = np.zeros(image.shape)
    i, j = np.where(image >= high_th)
    mask[i, j] = 255
    
    i, j = np.where((low_th <= image) & (image < high_th))
    mask[i, j] = 1
    return mask

def hysteresis(mask):
    new_mask = mask.copy()
    i = np.pi
    row, col = np.where(new_mask == 255)
    q = Queue()
    for ind in range(row.shape[0]):
        for i in range(row[ind] - 1, row[ind] + 2):
            for j in range(col[ind] - 1, col[ind] + 2):
                if new_mask[i, j] == 1:
                    new_mask[i, j] = 255
                    q.put((i, j))
    
    while not q.empty():
        r, c = q.get()
        for i in range(r - 1, r + 2):
            for j in range(c - 1, c + 2):
                if new_mask[i, j] == 1:
                    new_mask[i, j] = 255
                    q.put((i, j))
                    
    new_mask[new_mask == 1] = 0
    return new_mask

def canny_edge(image, low_th, high_th, sigma):
    image = np.float32(image)
    blurred_image = apply_guassian(image, 11, sigma)
    g, theta = apply_sobel(blurred_image)
    mask = non_maximum_suppresion(g, theta)
    mask = threshold(mask, low_th, high_th)
    e = hysteresis(mask)
    return e, blurred_image, g

cameraman = cv2.imread('cameraman.png', 0)
lena = cv2.imread('lena.png', 0)

low_th = 10
high_th = 100
sigma = 1
lena_sk_canny = feature.canny(lena, sigma = 1, low_threshold=low_th, high_threshold=high_th)
lena_my_canny, lena_blurred, lena_g = canny_edge(lena, low_th, high_th, sigma)

cameraman_sk_canny = feature.canny(cameraman, sigma = 1, low_threshold=low_th, high_threshold=high_th)
cameraman_my_canny, cameraman_blurred, cameraman_g = canny_edge(cameraman, low_th, high_th, sigma)


plt.figure()
plt.imshow(lena, cmap='gray')
plt.title('lena')
plt.show(block=False)

plt.figure()
plt.imshow(lena_blurred, cmap='gray')
plt.title('lena: blurred')
plt.show(block=False)

plt.figure()
plt.imshow(lena_g, cmap='gray')
plt.title('lena: Gradient')
plt.show(block=False)

plt.figure()
plt.imshow(lena_sk_canny, cmap='gray')
plt.title('lena: skimage canny')
plt.show(block=False)

plt.figure()
plt.imshow(lena_my_canny, cmap='gray')
plt.title('lena: my canny')
plt.show(block=False)


plt.figure()
plt.imshow(cameraman, cmap='gray')
plt.title('cameraman')
plt.show(block=False)

plt.figure()
plt.imshow(cameraman_blurred, cmap='gray')
plt.title('cameraman: blurred')
plt.show(block=False)

plt.figure()
plt.imshow(cameraman_g, cmap='gray')
plt.title('cameraman: Gradient')
plt.show(block=False)

plt.figure()
plt.imshow(cameraman_sk_canny, cmap='gray')
plt.title('cameraman: skimage canny')
plt.show(block=False)

plt.figure()
plt.imshow(cameraman_my_canny, cmap='gray')
plt.title('cameraman: my canny')
plt.show(block=True)



