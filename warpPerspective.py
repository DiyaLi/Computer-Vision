import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread('image.png')

# scaling
height, width = img.shape[:2]
rows,cols = img.shape[:2]
res = cv2.resize(img,(width*2,height*2))

# translation
# M=[1 0 tx
#    0 1 ty]
M = np.float32([[1,0,100],[0,1,50]])
dst = cv2.warpAffine(img,M,(width,height))

# rotation
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
rot = cv2.warpAffine(img,M,(cols,rows))

# # Affine Transformation
# M = cv2.getAffineTransform(pts1,pts2)

# dst = cv2.warpAffine(img,M,(cols,rows))

# # Perspective Transformation
# M = cv2.getPerspectiveTransform(pts1,pts2)

# dst = cv2.warpPerspective(img,M,(300,300))


# Plots
plt.figure(0)
plt.subplot(211)
plt.imshow(img,cmap = 'gray')
plt.title('Original')
plt.subplot(212)
plt.imshow(res,cmap = 'gray')
plt.title('Scaling')

plt.figure(1)
plt.subplot(211)
plt.imshow(img,cmap = 'gray')
plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(212)
plt.imshow(dst,cmap = 'gray')
plt.title('Translation')
plt.xticks([]), plt.yticks([])

plt.figure(2)
plt.subplot(211)
plt.imshow(img,cmap = 'gray')
plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(212)
plt.imshow(rot,cmap = 'gray')
plt.title('Rotation')
plt.xticks([]), plt.yticks([])

# plt.figure(3)
# plt.subplot(211)
# plt.imshow(img,cmap = 'gray')
# plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(212)
# plt.imshow(dst,cmap = 'gray')
# plt.title('Affine Transformation')
# plt.xticks([]), plt.yticks([])

# plt.figure(4)
# plt.subplot(211)
# plt.imshow(img,cmap = 'gray')
# plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(212)
# plt.imshow(dst,cmap = 'gray')
# plt.title('Perspective Transformation')
# plt.xticks([]), plt.yticks([])



plt.show()

