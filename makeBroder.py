import matplotlib.pyplot as plt
import numpy as np
import cv2

im = cv2.imread('image.png')
border = cv2.copyMakeBorder(im,20,20,20,20,cv2.BORDER_CONSTANT,value=[0,0,0])
border1 = cv2.copyMakeBorder(im,20,20,20,20,cv2.BORDER_WRAP)
border2 = cv2.copyMakeBorder(im,20,20,20,20,cv2.BORDER_REPLICATE)
border3 = cv2.copyMakeBorder(im,20,20,20,20,cv2.BORDER_REFLECT)

fig = plt.figure()
a=fig.add_subplot(2,2,1)
imgplot = plt.imshow(border)
a.set_title('constant')
plt.xticks([]), plt.yticks([])
a=fig.add_subplot(2,2,2)
imgplot = plt.imshow(border1)
a.set_title('wrap')
plt.xticks([]), plt.yticks([])
a=fig.add_subplot(2,2,3)
imgplot = plt.imshow(border2)
a.set_title('replicate')
plt.xticks([]), plt.yticks([])
a=fig.add_subplot(2,2,4)
imgplot = plt.imshow(border3)
a.set_title('reflect')
plt.xticks([]), plt.yticks([])
plt.show()

