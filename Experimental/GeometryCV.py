import numpy as np
import cv2
import os
import math
import matplotlib.pyplot as plt

'''img = cv2.imread('Messi.jpg', -1)
cols, rows, channels = img.shape
#res = cv2.resize(img, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
#res = cv2.getRotationMatrix2D((col/2, row/2), 65, 1)
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
dst = cv2.warpAffine(img,M,(cols,rows))

#cv2.imshow('img', img)
cv2.imshow('dst', dst)
cv2.waitKey(0)'''

img = cv2.imread("Sudoku.jpg", -1)

'''pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(300,300))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()

cv2.imshow('img', img)
cv2.imshow('dst', dst)'''

img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
'''th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)'''

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in xrange(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
