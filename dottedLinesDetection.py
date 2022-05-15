import cv2
import numpy as np

from scipy.ndimage import convolve1d
from scipy.signal import firwin, welch
import scipy as scipy

def remove_lines(image, low,high, num_taps=65, eps=0.025):
  """Removes horizontal line artifacts from scanned image.
  Args:
    image: 2D or 3D array.
    distortion_freq: Float, distortion frequency in cycles/pixel, or
      `None` to estimate from spectrum.
    num_taps: Integer, number of filter taps to use in each dimension.
    eps: Small positive param to adjust filters cutoffs (cycles/pixel).
  Returns:
    Denoised image.
  """
  image = np.asarray(image, float)
  print(low,high)
  bp = scipy.signal.firwin(85, [low, high], pass_zero=False,fs =1)
  #hpf = firwin(num_taps, distortion_freq - eps,
  #             pass_zero='highpass', fs=1)
  #bp = firwin(int(num_taps), [low,high] ,
  #             pass_zero=False, fs=1)

  lpf = scipy.signal.firwin(num_taps, eps, pass_zero='lowpass', fs=1)
  return  (convolve1d(image, bp, axis=0))

fmaxPixel =12 # minimum interval for dash repetation
fminPixel =7 # maximum interval for dash repetation

img = cv2.imread(os.path.join('images','test.png'))
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('imgGray', imgGray)
cv2.waitKey()

rows,cols = imgGray.shape
maskImage = np.full((rows, cols), 0, dtype=np.uint8)

kernelL = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
imgLaplacian = cv2.filter2D(imgGray, cv2.CV_32F, kernelL)
imgResult = imgLaplacian

imgResult = np.clip(imgResult, 0, 255)
imgResult = imgResult.astype('uint8')
imgLaplacian = imgResult
cv2.imshow('imgLaplacian', imgLaplacian)
cv2.waitKey()

bpfMin = 1/fmaxPixel
bpfMax = 1/fminPixel 
print(bpfMin,bpfMax)
maskImage = remove_lines(imgLaplacian,bpfMin,bpfMax)

maskImage1 = maskImage.copy()
max = np.max(maskImage1)
min = np.min(maskImage1)
maskImage1 = (maskImage1 -(max+min))*255/(max -min)
maskImage1 = np.clip(maskImage1, 0, 255)
maskImage1 = maskImage1.astype('uint8')
cv2.imshow('maskImage1', maskImage1)
cv2.waitKey()

imgLaplacian1 = cv2.filter2D(maskImage1, cv2.CV_32F, kernelL)
imgResult1 = imgLaplacian1
imgResult1[imgResult1 <0.7 * np.max(imgResult1)]= 0
imgResult1 = np.clip(imgResult1, 0, 255)
imgResult1 = imgResult1.astype('uint8')
imgLaplacian1 = imgResult1
cv2.imshow('imgLaplacian1', imgLaplacian1)
cv2.waitKey()

imgLines= cv2.HoughLinesP(imgLaplacian1,1,np.pi/180,10, minLineLength = 50, maxLineGap = 15)
x = np.array([imgLines[0][0][0],imgLines[0][0][2],imgLines[1][0][0],imgLines[1][0][2]])
y = np.array([imgLines[0][0][1],imgLines[0][0][3],imgLines[1][0][1],imgLines[1][0][3]])
bbXMin = np.min(x)
bbXMax = np.max(x)

bbYMin = np.min(y)
bbYMax = np.max(y)
print('bounding box',bbXMin,bbYMin,bbXMax,bbYMax)

# compensate for Laplacian shifts [twice] and misses in first and last dots in the HoughLinesP detection
cv2.rectangle(img, (bbXMin,bbYMin- (int((fminPixel+fmaxPixel)/2))), (bbXMax,bbYMax+ (int((fminPixel+fmaxPixel)/2)+ 2*3)), (255, 0, 0) , 2) 
cv2.imwrite('dashedLineDetectionx.png', img)

cv2.imshow('img', img)
cv2.waitKey()
cv2.imshow('maskImage', maskImage)
cv2.waitKey()