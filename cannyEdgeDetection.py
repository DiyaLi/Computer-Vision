import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('image.png',0)
blur = cv2.GaussianBlur(img,(5,5),3)

edges = cv2.Canny(img,100,200)
edges1 = cv2.Canny(blur,100,200)
sobel = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=3)

plt.figure(0)
plt.subplot(221),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(blur,cmap = 'gray')
plt.title('Blur Image'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(edges1,cmap = 'gray')
plt.title('Edge Blur'), plt.xticks([]), plt.yticks([])


# shift images
plt.figure(1)
left=np.copy(blur)
left[:,:-2]=blur[:,2:]
right=np.copy(blur)
right[:,2:]=blur[:,:-2]
Diff=np.float64(left)-np.float64(right)

plt.subplot(131),plt.imshow(left,cmap = 'gray')
plt.title('Left'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(Diff,cmap = 'gray')
plt.title('Blur Image'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(right,cmap = 'gray')
plt.title('Right'), plt.xticks([]), plt.yticks([])

# laplacian of Gaussian
plt.figure(2)
laplacian = cv2.Laplacian(img,cv2.CV_64F)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian Image'), plt.xticks([]), plt.yticks([])


plt.show()
