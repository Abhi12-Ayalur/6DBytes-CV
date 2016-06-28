import socket
from threading import *
import json
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

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
imageFrame = np.zeros((456, 507, 3), np.uint8)

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
    capComm= str("frame.jpg")
    os.system("fswebcam -r 507x456 --no-banner " + capComm)
    capImg= cv2.imread(capComm, -1)
    global n
    n = n+1
    return capImg

def showImage(capImg):
    cv2.imshow('img', capImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def colorAndCornerRecognition(capImg):
    # Get image, make it hsv, filter color of interest and de-noise
    hsv = cv2.cvtColor(capImg, cv2.COLOR_BGR2HSV)
    lower_color = np.array([40, 50, 50])
    upper_color = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    res = cv2.bitwise_and(capImg, capImg, mask=mask)
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

    # Find the appropriate contour
    while (existMoments == 0.0):
        print('Num of Contours ', len(contours))
        avgContours = int(len(contours)/2)
        for i in range (0, avgContours):
            print ("i is ", i, "avg is ", avgContours )
            cnt = contours[avgContours + i]
            M = cv2.moments(cnt)
            existMoments = float(M['m10'])
            if (existMoments != 0.0):

                cv2.drawContours(image, contours, 0, (0,255,0), 3)

                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                print ('cx is', cx, ' cy is ', cy)
                break

            cnt = contours[avgContours - i]
            M = cv2.moments(cnt)
            existMoments = float(M['m10'])
            if (existMoments != 0.0):

                cv2.drawContours(image, contours, 0, (0,255,0), 3)

                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                print ('cx is', cx, ' cy is ', cy)
                break
            iterations = 0
            iterations = iterations +1
            if (iterations > avgContours + 1):
                break
            

    #showImage(image)
    #area = cv2.contourArea(cnt)
    #print (area)
    if (M['m00'] == 0.0):
        cx = 320
        cy = 240

    global imageFrame
    imageFrame = cv2.rectangle(imageFrame, (cx -1, cy-1), (cx+1, cy+1), (255,0,0), -1, 8, 0)  
    return (cx, cy)
    cv2.imshow(imageFrame, 'frame')
    cv2.waitkey(0)
    
    
def computeImage(cx, cy):
    robotX = str((cx*507/640 -232)/1000)
    robotY = str(0.636)
    robotZ = str((cy*456/480 +10)/1000)
    return (robotX, robotY, robotZ)
    

def perform_robot_dance(client):
    limit = input("enter amount of times you want to capture: ")
    limit = int(limit)
    for i in range (0, limit):
        #while (1):
        capImg = captureAndLoadImage() 
        (cx, cy) = colorAndCornerRecognition(capImg)
        #(robotX, robotY, robotZ) = computeImage(cx, cy)
        #msg_to_robot = '[1000][3][' + robotX + '][' + robotY + '][' + robotZ + ']'
        #print (msg_to_robot)
        #client.sock.send(msg_to_robot.encode())
        #data = client.sock.recv(1024).decode()
        #print (data)

    
                 
# Real code
#startClientServerThreads()
perform_robot_dance(client)

