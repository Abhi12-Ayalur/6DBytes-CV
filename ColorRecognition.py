import cv2
import numpy as np

img = cv2.imread('imag42.jpg', -1)


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_green = np.array([30, 50, 50])
upper_green = np.array([80, 255, 255])

mask = cv2.inRange(hsv, lower_green, upper_green)

res = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow('frame', img)
cv2.imshow('mask',mask)
cv2.imshow('res',res)

cv2.waitKey(0)
cv2.destroyAllWindows()

