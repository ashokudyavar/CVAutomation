import numpy as np
import scipy.fftpack as fp
from skimage.io import imread
from skimage.color import rgb2gray, gray2rgb
import matplotlib.pylab as plt
from skimage.draw import rectangle_perimeter
import os
import cv2 as cv2

im = 255*rgb2gray(imread('home.png'))    # target
im2 = imread('home.png')

F = fp.fftn(im)                   

print(im2.shape)
directory = os.path.join(os.getcwd(),'segment')
for filename in os.listdir(directory):
    if filename.endswith(".png") or filename.endswith(".jpg"): 
        im_tm = 255*rgb2gray(imread(os.path.join(directory,filename))) # template
        print(filename)
        # FFT 
        F_tm = fp.fftn(im_tm, shape=im.shape)
    
        # compute the best match location
        F_cc = F * np.conj(F_tm)
        c = (fp.ifftn(F_cc/np.abs(F_cc))).real
        i, j = np.unravel_index(c.argmax(), c.shape)
        print(i, j)

        # 214 317
        #im2 = (gray2rgb(im)).astype(np.uint8)
        rr, cc = rectangle_perimeter((i,j), end=(i + im_tm.shape[0], j + im_tm.shape[1]), shape=im.shape)
        print((i + im_tm.shape[0], j + im_tm.shape[1]))
        print(min(rr),min(cc))
        for x in range(-2,2):
            for y in range(-2,2):
                im2[rr + x-1, cc + y-1] = (255,0,0)
        icon = filename.split('.')[-2]
        cv2.putText(im2,icon,(j + im_tm.shape[1],i + im_tm.shape[0]), cv2.FONT_HERSHEY_SIMPLEX,0.3,(209, 80, 0, 255),1) #font stroke

# show the output image
plt.figure(figsize=(50,50))
plt.imshow(im2)
plt.axis('off')
plt.savefig('foo.png')
plt.show()

