import socket
from threading import *
import json
import os
import cv2
import numpy as np
from matplotlib import pyplot as pl

cap = cv2.VideoCapture(0)

while(timeit.default_timer() < 30):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #cv2.imshow('frame',frame)
    cv2.waitKey(35)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
