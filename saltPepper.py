import matplotlib.pyplot as plt
import numpy as np
import cv2

im = cv2.imread('image.png')
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

# create noise
black=3
white=253
rmatrix = np.random.randint(256,size=(gray.shape[0],gray.shape[1]))
noiseImg = np.copy(gray)
noiseImg[rmatrix <= black] = 0
noiseImg[rmatrix >= white] = 255
rmnoise = cv2.medianBlur(noiseImg,5)

fig = plt.figure()
a=fig.add_subplot(1,2,1)
imgplot = plt.imshow(noiseImg,cmap='gray')
a.set_title('s&p noise')
plt.xticks([]), plt.yticks([])
a=fig.add_subplot(1,2,2)
imgplot = plt.imshow(rmnoise,cmap='gray')
a.set_title('remove noise')
plt.xticks([]), plt.yticks([])
plt.show()
