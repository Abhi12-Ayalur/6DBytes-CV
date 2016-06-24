import numpy as np
import cv2
import os
import math
os.system("fswebcam -r 507x456 --no-banner image11.jpg")

def showImage(capImg):
    cv2.imshow('img', capImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('image11.jpg',-1)
height, width, channel = img.shape
topy= height
topx = width

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_color = np.array([0,255,255])
upper_color = np.array([0,255,255])

mask = cv2.inRange(hsv, lower_color, upper_color)

res = cv2.bitwise_and(img,img, mask=mask)

'''def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x,y), 100, (255,255,255), -1)'''


'''cap = cv2.VideoCapture(-1)

while(True):

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('hjhj', gray)
    if cv2.waitKey(0) & 0xFF -- ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''

propx = (topx/512)
propy = (topy/512)

'''lineX1 = int(0*propx)
lineY2 = int(0*propy)
lineX2 = int(511*propx)
lineY1 = int(511*propy)

img = cv2.line(img, (lineX1,lineY1), (lineX2, lineY2), (255,255,255), 5)'''

w = 100*(propx+propy)/2
x1 = int(topx/2 - w/2)
x2 = int(topx/2 + w/2)
y1 = int(topy/2 + w/2)
y2 = int(topy/2 - w/2)

img = cv2.rectangle(res, (x1,y1), (x2,y2), (0,255,0),3)

img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
showImage(img) 

ret, thresh = cv2.threshold(img, 15, 250, 0)
showImage(thresh) 
image, contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#showImage(image)
cv2.drawContours(img, contours, 0, (0,255,0), 3)
showImage(img)

print('Num of Contours ', len(contours))

cnt = contours[0]
M = cv2.moments(cnt)
print (M)

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
area = cv2.contourArea(cnt)

print (cx)
print (cy)
print (area)

'''xCircle = 40*propx
xCircle = int(xCircle)
yCircle = xCircle
radCircle = xCircle

img = cv2.circle(img, (xCircle, yCircle), radCircle, (0,0,255),-1)

x3 = int(topx - 60*propx)
y3 = int(topy - 110*propy)
minAx = int(50*propx)
majAx = int(100*propy)

img = cv2.ellipse(img, (x3, y3), (minAx,majAx), 0, 0, 360, (0,150,255), -1)'''

'''pt1X = int(70*propx)
pt1Y = int(60*propy)
pt2X = int(154*propx)
pt2Y = int(23*propy)
pt3X = int(500*propx)
pt3Y = int(3*propy)'''

#pts = np.array([[pt1X, pt1Y], [pt2X, pt2Y], [pt3X, pt3Y]], np.int32)
#pts = pts.reshape((-1,1,2))
#img = cv2.polylines(img, [pts], True, (100,100,234))

#font = cv2.FONT_HERSHEY_SIMPLEX
#startPtX = int(240*propx)
#startPtY = int(240*propy)
#scale = 2*(propx + propy)/2
#cv2.putText(img, 'Apurva', (startPtX, startPtY), font, scale, (210, 80, 150), 4, cv2.LINE_AA)

#cv2.imshow("kl", img)

'''cv2.setMouseCallback('kl', draw_circle)'''

''''''

#cv2.imshow('frame', img)
#cv2.imshow('mask',mask)
cv2.imshow('res',res)

'''sd = img[130:200, 175:245]
img[20:90, 140:210]=sd
cv2.imshow("kl", img)'''

cv2.waitKey(0)
cv2.destroyAllWindows()

