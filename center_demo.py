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

def captureAndLoadImage(): # Captures video in image frme
    
    ret, capImg = cap.read()
    cv2.waitKey(100)
    return capImg

def showImage(capImg): # Simple function to display an image
    cv2.imshow('img', capImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sortArray(x, y): #Sorts x and y coordinates into avgX/Y arrays, appending it if it doesn't exist
    global avgX
    global avgY
    global p
    length = len(avgX)
    for a in range(0, length):
        if (abs(x - avgX[a]) < 50) and (abs(y - avgY[a]) < 50):
            avgX[a] = (x+avgX[a]*p[a])/(p[a]+1)
            avgY[a] = (y+avgY[a]*p[a])/(p[a]+1)
            p[a] = p[a] +1
            return
    avgX.append(x)
    avgY.append(y)
    p.append(1)
def checkForAverage(p): #Checks for the average value with the greatest number of occurences
    global maxVal
    global b
    maxVal = 0
    for a in range(0, len(p)):
        val = p[a]
        if (val > maxVal):
            maxVal = val
            b = a
    return b
            

def colorAndCornerRecognition(capImg): # Uses color and contour recognition to determine the points of an object in the image
    # Call global variables, define local variables
    global image
    global contours
    global M
    global cnt
    global existMoments
    global iterations
    global avgX
    global avgY
    global p
    global posX
    global posY
    global mx
    global my
    global samples
    existMoments = 0.0

    # Get image, make it hsv, filter color of interest 
    hsv = cv2.cvtColor(capImg, cv2.COLOR_BGR2HSV)
    lower_color = np.array([5, 100, 160])
    upper_color = np.array([20, 225, 230])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    cv2.imshow('mask', mask)
    res = cv2.bitwise_and(capImg, capImg, mask=mask)
    capImg = res

    # Convert to grayscale and create thresholds to find contours
    img = cv2.cvtColor(capImg,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img, 15, 250, cv2.THRESH_BINARY)
    image, contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow('thresh', image)
    # Find the appropriate contours, check the moments and get the average
    for i in range (0, len(contours)):
        #print ( 'contour len', len(contours), 'i ', i, 'Cluster len ', len(avgX))
        cnt = contours[i]
        #print ('contours ', contours)
        M = cv2.moments(cnt)
        existMoments = float(M['m10'])
        if (existMoments != 0.0):
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
            posX.append(x)
            posY.append(y)
            sortArray(x, y)
            

    if ( len(posX) <= 0 ):
        return (0, 0)
    
    b = checkForAverage(p)
    #print ('posX ', posX, ' posY ', posY, 'avgX ', avgX, 'avgY ', avgY, 'p ', p, 'b ', b)
    cx = avgX[b]
    cy = avgY[b]

    # Check mx/my thresholds to create existence of a possible point        
    if (abs(cx - mx) < 50) and (abs(cy-my) < 50):
        samples = samples + 1
        #print ('Within threshold cx is', cx, ' cy is ', cy, ' mx', mx, ' my', my,
        #       ' gx', gx, ' gy', gy, ' samples', samples)
        cx = mx
        cy = my
        
    else:
        samples = 0
        #print ('Moved cx is', cx, ' cy is ', cy, ' mx', mx, ' my', my,
        #      ' gx', gx, ' gy', gy, ' samples', samples)
        if (((abs(gx - cx)<50 and (abs(gy-cy) <50)))):
            mx = gx
            my = gy
        else:
            mx = cx
            my = cy
            
    # Print supposed cx       
    #print ( 'contour len', len(contours), 'i ', i, 'Cluster len ', len(avgX))
    #print ('cx ', cx, 'cy ' , cy)
    return (cx, cy)
    
    
def computeImage(cx, cy): # Compute proportions and coordinates for movement
    robotX = ((cx*507/640 -232)/1000)
    robotY = (0.636)
    robotZ = ((466 - cy*456/480)/1000)
    return (robotX, robotY, robotZ)
    
def moveRobot(robotX, robotZ): # Computes movement for actual robot, coordinates are based on robot constraints
    moveY = 0.0
    print ('robotX ', robotX, 'robotZ ', robotZ)
    if (abs(robotX*1000 - 476/2) > 35):
        moveX = float(robotX*1000 - 476/2)
        print ('moveX1 ' , moveX)
        robotEndX = moveX + robotX*1000
    else:
        moveX = 0
        print ('moveX2 ' , moveX)
    if (abs(robotZ*1000 - 43/2) > 35):
        moveZ = float(robotZ*1000 - 43/2)
        print ('moveZ1 ', moveZ)
        robotEndZ = moveZ + robotZ*1000
    else:
        moveZ = 0
        print ('moveZ2 ', moveZ)
    #if (robotEndX > 466) or (robotEndX < 10) or (robotEndZ < -232) or (robotEndZ > 275):
        #moveX = 0
        #moveZ = 0
    print ('moveX ' , moveX, 'moveY ', moveY, 'moveZ ', moveZ)
    return (moveX, moveY, moveZ)

def moveSimRobot(robotX, robotZ): # Puts dots that image would move to in a simulator
    moveY = 0
    inCenter = 0
    if (abs(int(robotX - 507/2)) > 50):
        moveX = int(robotX)
        inCenter = 1
    else:
        moveX = int(507/2)
    if (abs(int(robotZ - 456/2)) > 50):
        moveZ = int(robotZ)
        inCenter = 1
    else:
        moveZ = int(456/2)
    return (moveX, moveY, moveZ, inCenter)

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
    start = timeit.default_timer()
    stop = start

    while (stop - start < 15):
        b = 0
        avgX=[]
        avgY=[]
        p =[]
        posX= []
        posY = []
        
        capImg = captureAndLoadImage()
        (cx, cy) = colorAndCornerRecognition(capImg)
        
        if (samples >= 2) and (cx != 0):
            global imageFrame
            (robotX, robotY, robotZ) = computeImage(cx, cy)
            print ('robotX ' , robotX , ' robotY ' , robotY , ' robotZ ' , robotZ)
            '''robotX = str(robotX)
            robotY = str(robotY)
            robotZ = str(robotZ)'''
            (moveX, moveY, moveZ) = moveRobot(robotX, robotZ)
            moveX = str(moveX/1000)
            moveY = str(moveY/1000)
            moveZ = str(moveZ/1000)
            #cx = int(cx*507/640)
            #cy = int(cy*456/480)
            #(moveSimX, moveSimY, moveSimZ, inCenter) = moveSimRobot(cx, cy)
           # if (inCenter != 0):
           #     imageFrame = cv2.rectangle(imageFrame, (moveSimX -1, moveSimZ-1), (moveSimX+1, moveSimZ+1), (255,0,0), -1, 8, 0)  
            msg_to_robot = '[1001][3][' + moveX + '][' + moveY + '][' + moveZ + ']'
            print (msg_to_robot)
            client.sock.send(msg_to_robot.encode())
            #data = client.sock.recv(1024).decode()
            #print (data)
            #return (cx, cy)
            #print ('Printing dot cx is', cx, ' cy is ', cy, ' mx', mx, ' my', my,
            #       ' gx', gx, ' gy', gy, ' samples', samples)
            #cv2.imshow('frame', imageFrame)
            gx = mx
            gy = my
        else:
            if (((abs(gx - cx)<50 and (abs(gy-cy) <50)))):
                if (not((abs(mx - cx)<50 and (abs(my-cy) <50)))):
                    mx = gx
                    my = gy
                    samples = 0
        stop = timeit.default_timer()
        #print ('time take is: ' , stop - start)
        #print ('Printing dot cx is', cx, ' cy is ', cy, ' mx', mx, ' my', my,
                 #  ' gx', gx, ' gy', gy, ' samples', samples)
    print ('Done')
    showImage(imageFrame)
                 
#Real code
startClientServerThreads()
#perform_robot_dance(client)

