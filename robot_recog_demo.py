import socket
from threading import *
import json
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import timeit

# Declare socket parameters
host = "192.168.1.84"
port = 60000
print (host)
print (port)
n = 1
global existMoments
existMoments = float(0.0)
global image
global M
global cx
global cy
global imageFrame
global samples
global mx
global my
global gx
global gy
global maxVal
global avgX
global avgY
global p
global posX
global posY
global b
maxVal = 0
mx = 0
my = 0
gx = 0
gy = 0 
samples = 0
b = 0
avgX=[]
avgY=[]
p =[]
posX= []
posY = []
imageFrame = np.zeros((456, 507, 3), np.uint8)
cap = cv2.VideoCapture(0)

class client(Thread):
    def __init__(self, socket, address):
        Thread.__init__(self)
        self.sock = socket
        self.addr = address
        self.start()

    def run(self):
        print('Client connected\n')
        msg_from_robot = self.sock.recv(1024).decode()
        print('Robot sent:', msg_from_robot)
        while 1:
            perform_robot_dance(self)
            #self.sock.close()

def startClientServerThreads():
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind((host, port))
    serversocket.listen(5)
    print ('server started and listening')
    try:
        while 1:
            clientsocket, address = serversocket.accept()
            client(clientsocket, address)
    except (KeyboardInterrupt, SystemExit):
        sys.exit()

def captureAndLoadImage():
    ret, capImg = cap.read()
    #cv2.imshow('frame',frame)
    cv2.waitKey(35)
    #global n
    #n = n+1
    return capImg

def showImage(capImg):
    cv2.imshow('img', capImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sortArray(x, y):
    global avgX
    global avgY
    global p
    length = len(avgX)
    for a in range(0, length):
        if (abs(x - avgX[a]) < 35) and (abs(y - avgY[a]) < 35):
            #print ('calculating average')
            avgX[a] = (x+avgX[a]*p[a])/(p[a]+1)
            avgY[a] = (y+avgX[a]*p[a])/(p[a]+1)
            p[a] = p[a] +1
            return
    avgX.append(x)
    avgY.append(y)
    p.append(1)
    #print ('new average created')

def checkForAverage(p):
    global maxVal
    for a in range(0, len(p)):
        val = p[a]
        if (val > maxVal):
            maxVal = val
            p[b] = p[a]
    return b
            

def colorAndCornerRecognition(capImg):
    # Get image, make it hsv, filter color of interest and de-noise
    hsv = cv2.cvtColor(capImg, cv2.COLOR_BGR2HSV)
    lower_color = np.array([40, 50, 50])
    upper_color = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    res = cv2.bitwise_and(capImg, capImg, mask=mask)
    #cv2.imshow('res', res)
    #cv2.waitKey(100)
    capImg = res
    #capImg = cv2.fastNlMeansDenoisingColored(capImg, None, 10, 10, 7, 21)

    # Convert to grayscale and create thresholds to find contours
    global image
    global contours
    global M
    global cnt
    global existMoments
    global iterations
    existMoments = 0

    img = cv2.cvtColor(capImg,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img, 15, 250, cv2.THRESH_BINARY)
    image, contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    global avgX
    global avgY
    global p
    global posX
    global posY
    # Find the appropriate contour
    for i in range (0, len(contours)):
        #print ( 'contour len', len(contours), 'i ', i, 'Cluster len ', len(avgX))
        cnt = contours[i]
        M = cv2.moments(cnt)
        existMoments = float(M['m10'])
        if (existMoments != 0.0):
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
            posX.append(x)
            posY.append(y)
            sortArray(x, y)                
    b = checkForAverage(p)
    cx = avgX[b]
    cy = avgY[b]
            
            
    print('Num of Contours ', len(contours))
        #avgContours = int(len(contours)/2)
    '''for i in range (0, avgContours):
            #print ("i is ", i, "avg is ", avgContours )
            cnt = contours[avgContours + i]
            M = cv2.moments(cnt)
            existMoments = float(M['m10'])
            if (existMoments != 0.0):

                cv2.drawContours(image, contours, 0, (0,255,0), 3)
                #cv2.imshow('contours', image)
                #cv2.waitKey(200)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                break
            iterations = 0
            iterations = iterations +1
            if (iterations > avgContours + 1):
                break'''
    global mx
    global my
    global samples
    if (abs(cx - mx) < 35) and (abs(cy-my) < 35):
        samples = samples + 1
        #print ('Within threshold cx is', cx, ' cy is ', cy, ' mx', mx, ' my', my,
        #       ' gx', gx, ' gy', gy, ' samples', samples)
        cx = mx
        cy = my
        
    else:
        samples = 0
        #print ('Moved cx is', cx, ' cy is ', cy, ' mx', mx, ' my', my,
        #      ' gx', gx, ' gy', gy, ' samples', samples)
        if (((abs(gx - cx)<35 and (abs(gy-cy) <35)))):
            mx = gx
            my = gy
        else:
            mx = cx
            my = cy
            
            
            

    #showImage(image)
    #area = cv2.contourArea(cnt)
    #print (area)
    #if (M['m00'] == 0.0):
      #  cx = 320
      #  cy = 240

    #global imageFrame
    #imageFrame = cv2.rectangle(imageFrame, (cx -1, cy-1), (cx+1, cy+1), (255,0,0), -1, 8, 0)
    
    print ( 'contour len', len(contours), 'i ', i, 'Cluster len ', len(avgX))
    print ('cx ', cx, 'cy ' , cy)
    return (cx, cy)
    #cv2.imshow(imageFrame, 'frame')
    #cv2.waitkey(0)
    
    
def computeImage(cx, cy):
    robotX = str((cx*507/640 -232)/1000)
    robotY = str(0.636)
    robotZ = str((466 - cy*456/480)/1000)
    return (robotX, robotY, robotZ)
    

def perform_robot_dance(client):
    global gx
    global gy
    global mx
    global my
    global samples
    global maxVal
    global avgX
    global avgY
    global p
    global posX
    global posY
    global b
    #limit = input("enter amount of times you want to capture: ")
    #limit = int(limit)
    start = timeit.default_timer()
    stop = start
    while (stop - start < 15):
        b = 0
        avgX=[]
        avgY=[]
        p =[]
        posX= []
        posY = []
        
        #while (1):
        capImg = captureAndLoadImage()
        (cx, cy) = colorAndCornerRecognition(capImg)
        
        if (samples > 2):
            global imageFrame
            (robotX, robotY, robotZ) = computeImage(cx, cy)
            cx = int(cx*507/640)
            cy = int(cy*456/480)
            imageFrame = cv2.rectangle(imageFrame, (cx -1, cy-1), (cx+1, cy+1), (255,0,0), -1, 8, 0)  
            msg_to_robot = '[1000][3][' + robotX + '][' + robotY + '][' + robotZ + ']'
            print (msg_to_robot)
            #client.sock.send(msg_to_robot.encode())
            #data = client.sock.recv(1024).decode()
            #print (data)
            #return (cx, cy)
            cv2.imshow('frame', imageFrame)
            print ('Printing dot cx is', cx, ' cy is ', cy, ' mx', mx, ' my', my,
                   ' gx', gx, ' gy', gy, ' samples', samples)

            gx = mx
            gy = my
        else:
            if (((abs(gx - cx)<35 and (abs(gy-cy) <35)))):
                if (not((abs(mx - cx)<35 and (abs(my-cy) <35)))):
                    mx = gx
                    my = gy
                    samples = 0
        stop = timeit.default_timer()
        #print ('time take is: ' , stop - start)
        #print ('Printing dot cx is', cx, ' cy is ', cy, ' mx', mx, ' my', my,
                 #  ' gx', gx, ' gy', gy, ' samples', samples)
    print ('Done')
    cv2.waitKey(0)
    
                 
# Real code
#startClientServerThreads()
perform_robot_dance(client)

