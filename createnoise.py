import matplotlib.pyplot as plt
import numpy as np
import cv2

im = cv2.imread('image.png')
imf = cv2.GaussianBlur(im,(31,31),5)
imf1 = cv2.GaussianBlur(im,(31,31),50)
imf2 = cv2.GaussianBlur(im,(63,63),5)
imf3 = cv2.GaussianBlur(im,(63,63),50)

fig = plt.figure()
a=fig.add_subplot(2,2,1)
imgplot = plt.imshow(imf)
a.set_title('(31,31),5')
a=fig.add_subplot(2,2,2)
imgplot = plt.imshow(imf1)
a.set_title('(31,31),50')
a=fig.add_subplot(2,2,3)
imgplot = plt.imshow(imf2)
a.set_title('(63,63),5')
a=fig.add_subplot(2,2,4)
imgplot = plt.imshow(imf3)
a.set_title('(63,63),50')
plt.show()