import matplotlib.pyplot as plt
import numpy as np
import cv2

im = cv2.imread('1.jpg')
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

#remove noise 
img = cv2.GaussianBlur(gray,(3,3),0)

####################################################################################
# sobel operator
####################################################################################
# filter 2d function
k_xr = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])/8.
sobel_xr = cv2.filter2D(img,-1,k_xr)

k_xl = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])/8.
sobel_xl = cv2.filter2D(img,-1,k_xl)

k_yu = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])/8.
sobel_yu = cv2.filter2D(img,-1,k_yu)

k_yd = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])/8.
sobel_yd = cv2.filter2D(img,-1,k_yd)

k_ul = np.sqrt(k_xl**2+k_yu**2)
sobel_ul = cv2.filter2D(img,-1,k_ul)

k_ur = np.sqrt(k_xr**2+k_yu**2)
sobel_ur = cv2.filter2D(img,-1,k_ur)

k_dl = np.sqrt(k_xl**2+k_yd**2)
sobel_dl = cv2.filter2D(img,-1,k_dl)

k_dr = np.sqrt(k_xr**2+k_yd**2)
sobel_dr = cv2.filter2D(img,-1,k_dr)

# plot the results
plt.figure(1)
plt.subplot(3,3,1),plt.imshow(sobel_dr,cmap = 'gray')
plt.title('sobel_dr'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,2),plt.imshow(sobel_yd,cmap = 'gray')
plt.title('sobel_yd'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,3),plt.imshow(sobel_dl,cmap = 'gray')
plt.title('sobel_dl'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,4),plt.imshow(sobel_xr,cmap = 'gray')
plt.title('sobel_xr'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,5),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,6),plt.imshow(sobel_xl,cmap = 'gray')
plt.title('sobel_xl'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,7),plt.imshow(sobel_ur,cmap = 'gray')
plt.title('sobel_ur'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,8),plt.imshow(sobel_yu,cmap = 'gray')
plt.title('sobel_yu'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,9),plt.imshow(sobel_ul,cmap = 'gray')
plt.title('sobel_ul'), plt.xticks([]), plt.yticks([])
plt.suptitle('cv2.filter2D')

####################################################################################
# opencv function
# choose cv2.CV_64F to keep output data since some slope are negative (cv2.CV_8U )
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobel = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=3)

# plot the results
plt.figure(2)
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobel,cmap = 'gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])
plt.suptitle('cv2.Sobel')



############################################################################################
# show gradient image
from skimage.feature import hog
from skimage import data, color, exposure

fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

plt.figure(3)
plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(hog_image,cmap = 'gray')
plt.title('hog_image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(hog_image_rescaled,cmap = 'gray')
plt.title('hog_image_rescaled'), plt.xticks([]), plt.yticks([])

plt.show()

