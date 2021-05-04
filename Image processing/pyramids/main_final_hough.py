from cv2 import cv2
import numpy as np

img = cv2.imread('hti.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

canny_thr1 = 50
canny_thr2 = 100
edges = cv2.Canny(gray, threshold1 = canny_thr1, threshold2 = canny_thr2)

rho_res = 1
theta_res = np.pi / 180
lines = cv2.HoughLinesP(edges, rho_res, theta_res, 200, minLineLength=200)

print(lines.shape)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0 ,0 ,255), 1)

cv2.imwrite('lines.jpg', img)
