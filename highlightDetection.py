# import the necessary packages
import numpy as np
import argparse
import imutils
import glob
import cv2
from matplotlib import pyplot as plt
import heapq

img = cv2.imread('0014.png')

imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('img',img)
cv2.imshow('imgGray',imgGray)

# for i in range(1,10):
    # img_temp = img.copy()
    # unique, counts = np.unique(img_temp.reshape(-1, 3), axis=0, return_counts=True)
    # print(np.sort(np.unique(counts)))
    # secondL = np.sort(np.unique(counts))[-i]
    # secondIndex = int(np.where(counts == secondL)[0])
    # print('secondIndex',secondIndex)
    # print(np.argmax(counts))
    # print('color',unique[secondIndex])
    # img_temp[:,:,0], img_temp[:,:,1], img_temp[:,:,2] = unique[secondIndex]
    # cv2.imshow('img_temp' + str(i),img_temp)
    # cv2.waitKey(0)

img_temp = img.copy()
unique, counts = np.unique(img_temp.reshape(-1, 3), axis=0, return_counts=True)
print(np.sort(np.unique(counts)))
secondL = np.sort(np.unique(counts))[-2]
secondIndex = int(np.where(counts == secondL)[0])
print('secondIndex',secondIndex)
print(np.argmax(counts))
print('color',unique[secondIndex])
color2= unique[secondIndex]
#img_temp[img_temp == color2][:,:,0],img_temp[img_temp == color2][:,:,1],img_temp[img_temp == color2][:,:,2], = unique[secondIndex]
#print(img_temp[img_temp[:,:] == color2])
#img_temp[img_temp != color2] = [0,0,0]
img_temp[:,:,0], img_temp[:,:,1], img_temp[:,:,2] = unique[secondIndex]
img_temp1 = img.copy()
lower = np.array(color2)
upper = np.array(color2)
shapeMask = cv2.inRange(img_temp1, lower, upper)
cv2.imshow("obj shapeMask", shapeMask)

contours, hierarchy = cv2.findContours(shapeMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#cv2.imshow('dest_or',dest_or)

x=[]
y=[]
print(len(contours[0]))
for i in range(len(contours[0])):
    print('contour is ',contours[0][i])
    for x1,y1 in contours[0][i]:
        x.append(int(x1))
        y.append(int(y1))

print(x,y)
bbXMin = np.min(x)
bbXMax = np.max(x)
bbYMin = np.min(y)
bbYMax = np.max(y)

img3 = img.copy()
cv2.rectangle(img3, (bbXMin,bbYMin), (bbXMax,bbYMax), (0, 0, 255) , 2) 
cv2.imshow('img3',img3)
print(contours)
cv2.imwrite('SavehighlightDetection.png',img3)

cv2.waitKey(0)
