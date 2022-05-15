from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
import imutils
import os 

rng.seed(12345)
parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to input image.', default='home.png')
args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.input))
src1 = src.copy()
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
# Show source image
cv.imshow('Source Image', src)
# Show output image
cv.imshow('Black Background Image', src)

kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
imgLaplacian = cv.filter2D(src, cv.CV_32F, kernel)

sharp = np.float32(src)
imgResult =  imgLaplacian

imgLaplacian = cv.filter2D(src, cv.CV_32F, kernel)
cv.imshow('recent', imgLaplacian)
cv.waitKey()

# convert back to 8bits gray scale
imgResult = np.clip(imgResult, 0, 255)
imgResult = imgResult.astype('uint8')
imgLaplacian = np.clip(imgLaplacian, 0, 255)
imgLaplacian = np.uint8(imgLaplacian)
#cv.imshow('Laplace Filtered Image', imgLaplacian)
cv.imshow('New Sharped Image', imgResult)
cv.waitKey()
bw = cv.cvtColor(imgResult, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(bw, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
cv.imshow('Binary Image', thresh)
cv.waitKey()

laplacian = cv.Laplacian(thresh,cv.CV_8UC1) # Laplacian Edge Detection
minLineLength = 20
maxLineGap = 30
lines = cv.HoughLinesP(laplacian,1,np.pi/180,100,minLineLength,maxLineGap)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv.line(thresh,(x1,y1),(x2,y2),(0,0,0),3)
cv.imshow('no lines', thresh)
cv.waitKey()

kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_GRADIENT,kernel, iterations = 3)
cv.imshow('opening', opening)
cv.waitKey()


gray = cv.bilateralFilter(opening, 11, 17, 17)
edges = cv.Canny(opening, 10, 100)
# Find Contours
cnts = cv.findContours(opening.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv.contourArea, reverse = True)
print(len(cnts))
cv.drawContours(src, cnts, -1, (0, 255, 0), 3) 
  
cv.imshow('Contours', src) 
cv.waitKey(0) 
cv.imwrite('segments.png',src) 

for i in range(len(cnts)):
    x, y, width, height = cv.boundingRect(cnts[i])
    roi = src1[y:y+height, x:x+width]
    cv.imwrite(os.path.join(os.getcwd(),'segment',str(i)+"roi.png"), roi)

# ret, markers = cv.connectedComponents(sure_fg)
# # Add one to all labels so that sure background is not 0, but 1
# markers = markers+1
# # Now, mark the region of unknown with zero
# markers[unknown==255] = 0

# markers = cv.watershed(src,markers)
# src[markers == -1] = [255,0,0]

# #mark = np.zeros(markers.shape, dtype=np.uint8)
# mark = markers.astype('uint8')
# mark = cv.bitwise_not(mark)
# # uncomment this if you want to see how the mark
# # image looks like at that point
# cv.imshow('Markers_v2', mark)

# cv.waitKey()
# dist = cv.distanceTransform(bw, cv.DIST_L2, 3)
# # Normalize the distance image for range = {0.0, 1.0}
# # so we can visualize and threshold it
# cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
# cv.imshow('Distance Transform Image', dist)
# cv.waitKey()
# _, dist = cv.threshold(dist, 0.4, 1.0, cv.THRESH_BINARY)
# # Dilate a bit the dist image
# kernel1 = np.ones((3,3), dtype=np.uint8)
# dist = cv.dilate(dist, kernel1)
# cv.imshow('Peaks', dist)
# dist_8u = dist.astype('uint8')
# # Find total markers
# contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# # Create the marker image for the watershed algorithm
# markers = np.zeros(dist.shape, dtype=np.int32)
# # Draw the foreground markers
# for i in range(len(contours)):
    # cv.drawContours(markers, contours, i, (i+1), -1)
# # Draw the background marker
# cv.circle(markers, (5,5), 3, (255,255,255), -1)
# markers_8u = (markers * 10).astype('uint8')
# cv.imshow('Markers', markers_8u)
# cv.watershed(imgResult, markers)
# #mark = np.zeros(markers.shape, dtype=np.uint8)
# mark = markers.astype('uint8')
# mark = cv.bitwise_not(mark)
# # uncomment this if you want to see how the mark
# # image looks like at that point
# #cv.imshow('Markers_v2', mark)
# # Generate random colors
# colors = []
# for contour in contours:
    # colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
# # Create the result image
# dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
# # Fill labeled objects with random colors
# for i in range(markers.shape[0]):
    # for j in range(markers.shape[1]):
        # index = markers[i,j]
        # if index > 0 and index <= len(contours):
            # dst[i,j,:] = colors[index-1]
# # Visualize the final image
# cv.imshow('Final Result', dst)
# cv.waitKey()
