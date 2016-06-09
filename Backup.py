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

class client(Thread):
    def __init__(self, socket, address):
        Thread.__init__(self)
        self.sock = socket
        self.addr = address
        self.start()

    def run(self):
        while 1:
            print('Client connected\n')
            msg_from_robot = self.sock.recv(1024).decode()
            print('Robot sent:', msg_from_robot)
            perform_robot_dance()
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
    #showImage(capImg) 
   
    #capImg = cv2.fastNlMeansDenoisingColored(capImg, None, 10, 10, 7, 21)
    #showImage(capImg) 

    plt.imshow(capImg), plt.show()

    hsv = cv2.cvtColor(capImg, cv2.COLOR_BGR2HSV)

    lower_color = np.array([40, 50, 50])
    upper_color = np.array([80, 255, 255])

    mask = cv2.inRange(hsv, lower_color, upper_color)

    res = cv2.bitwise_and(capImg, capImg, mask=mask)

    capImg = res
    #showImage(capImg) 


    capImg = cv2.fastNlMeansDenoisingColored(capImg, None, 10, 10, 7, 21)
    #showImage(capImg) 
    
    #gray = cv2.cvtColor(capImg,cv2.COLOR_BGR2GRAY)
    #surf = cv2.xfeatures2d.SURF_create(1000)
    #kp, des = surf.detectAndCompute(capImg, None) #finds keypoints and descriptors in capImg
    #print (kp)
    #capImg = cv2.drawKeypoints(capImg, kp, None, (255,0,0), 4)

    while (existMoments == 0.0):
    
        #cv2.drawContours(image, contours, 0, (0,255,0), 3)
        #showImage(capImg)
        global image
        #capImg = cv2.copyMakeBorder(capImg,10,10,10,10,cv2.BORDER_CONSTANT,value = [255,255,255])

        img = cv2.cvtColor(capImg,cv2.COLOR_BGR2GRAY)
        #showImage(img) 

        ret, thresh = cv2.threshold(img, 15, 250, cv2.THRESH_BINARY)
        #showImage(thresh) 
        image, contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        print('Num of Contours ', len(contours))
        global cnt

        #print (contours)
        '''for x in range (0, len(contours)):
            cnt = contours[x]
            #print (cnt)
            global M
            M = cv2.moments(cnt)


            global existMoments


            existMoments = float(M['m10'])
            if (existMoments != 0.0):

                cv2.drawContours(image, contours, 0, (0,255,0), 3)

                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                #print (M)
                print ('cx is', cx, ' cy is ', cy)'''

        global contours
        avgContours = len(contours)/2
        for i in range (0, avgContours):
            print ("i is ", i, "avg is ", avgContours )
            cnt = contours[avgContours + i]
            global M
            M = cv2.moments(cnt)
            global existMoments
            existMoments = float(M['m10'])
            if (existMoments != 0.0):

                cv2.drawContours(image, contours, 0, (0,255,0), 3)

                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                print ('cx is', cx, ' cy is ', cy)
                break

            cnt = contours[avgContours - i]
            global M
            M = cv2.moments(cnt)
            existMoments = float(M['m10'])
            if (existMoments != 0.0):

                cv2.drawContours(image, contours, 0, (0,255,0), 3)

                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                print ('cx is', cx, ' cy is ', cy)
                break



    #capImg = cv2.fastNlMeansDenoisingColored(capImg, None, 15, 15, 7, 31)
    
    
    showImage(image)
    #cx = int(M['m10']/M['m00'])
    #cy = int(M['m01']/M['m00'])
    #area = cv2.contourArea(cnt)

    #capImg = image
    #showImage(capImg)
    #print (M)
    #print (cx)
    #print (cy)
    #print (area)
    
    '''corners = cv2.goodFeaturesToTrack(gray,2,0.01,10)
    corners = np.int0(corners)'''
    '''for i in contours:
        x,y, z = i.ravel()
        cv2.circle(capImg,(x,y),30,(0,0,255),-1)
    showImage(capImg)'''

    
def computeImage(capImg):

    return
    

def perform_robot_dance():
    limit = input("enter amount of times you want to capture: ")
    limit = int(limit)
    for i in range (0, limit):
        #while (1):
        capImg = captureAndLoadImage() 
        colorAndCornerRecognition(capImg)
        #computeImage(capImg)
        msg_to_robot = '[1000][3][0.270][0.635][0.020]'
        #self.sock.send(msg_to_robot.encode())
        print (msg_to_robot)

    
                 
# Real code
#startClientServerThreads()
perform_robot_dance()
