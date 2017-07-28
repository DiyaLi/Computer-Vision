import numpy as np
import cv2
from matplotlib import pyplot as plt

###################################################################################################
# # one dimension example
# x = np.random.randint(25,100,25)
# y = np.random.randint(175,255,25)
# z = np.hstack((x,y))
# z = z.reshape((50,1))
# z = np.float32(z)
# plt.figure(0)
# plt.subplot(121)
# plt.hist(z,256,[0,256])


# # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# # Set flags (Just to avoid line break in the code)
# flags = cv2.KMEANS_RANDOM_CENTERS

# # Apply KMeans
# compactness,labels,centers = cv2.kmeans(z,2,criteria,10,flags)

# A = z[labels==0]
# B = z[labels==1]

# # Now plot 'A' in red, 'B' in blue, 'centers' in yellow
# plt.subplot(122)
# plt.hist(A,256,[0,256],color = 'r')
# plt.hist(B,256,[0,256],color = 'b')
# plt.hist(centers,32,[0,256],color = 'y')
# plt.show()

##########################################################################################################################
img = cv2.imread('road2.png')
# reshape, -1 means that the number is unknown and the function will figure it out. In here, row number is unknown
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

# A=label.flatten()
# A[label.flatten()>0]=1
# new = center[A]
# new2=new.reshape((img.shape))

cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()