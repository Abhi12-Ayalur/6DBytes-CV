import cv2
events = [i for i in dir(cv2) if 'EVENT' in i]
print (events)
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print (flags)
