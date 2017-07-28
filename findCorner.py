import numpy as np
import cv2
from matplotlib import pyplot as plt

##################################################################################################################################
# Shi-Tomasi Corner Detector
##################################################################################################################################
# img = cv2.imread('road.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
# corners = np.int0(corners)

# for i in corners:
#     x,y = i.ravel()
#     cv2.circle(img,(x,y),3,255,-1)

# plt.imshow(img),plt.show()

##################################################################################################################################
# Harris Corner Detector
##################################################################################################################################
# filename = 'road.jpg'
# img = cv2.imread(filename)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray,2,3,0.04)

# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)

# # Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()]=[0,0,255]

# cv2.imshow('dst',img)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()

##################################################################################################################################
# fast feature detector
##################################################################################################################################
# img = cv2.imread('road.jpg',0)

# # Initiate FAST object with default values
# fast = cv2.FastFeatureDetector()

# # find and draw the keypoints
# kp = fast.detect(img,None)
# img2 = cv2.drawKeypoints(img, kp, color=(255,0,0))

# # Print all default params
# # print "Threshold: ", fast.getInt('threshold')
# # print "nonmaxSuppression: ", fast.getBool('nonmaxSuppression')
# # print "neighborhood: ", fast.getInt('type')
# # print "Total Keypoints with nonmaxSuppression: ", len(kp)

# cv2.imwrite('fast_true.png',img2)

# # Disable nonmaxSuppression
# fast.setBool('nonmaxSuppression',0)
# kp = fast.detect(img,None)

# print "Total Keypoints without nonmaxSuppression: ", len(kp)

# img3 = cv2.drawKeypoints(img, kp, color=(255,0,0))

# cv2.imwrite('fast_false.png',img3)

# plt.subplot(311)
# plt.imshow(img,cmap = 'gray')
# plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(312)
# plt.imshow(img2,cmap = 'gray')
# plt.title('Total Keypoints with nonmaxSuppression')
# plt.xticks([]), plt.yticks([])
# plt.subplot(313)
# plt.imshow(img3,cmap = 'gray')
# plt.title('Total Keypoints without nonmaxSuppression')
# plt.xticks([]), plt.yticks([])
# plt.show()

##################################################################################################################################
# SIFT (Scale-Invariant Feature Transform)
##################################################################################################################################

img = cv2.imread('road.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
img=cv2.drawKeypoints(gray,kp,img)
cv2.imwrite('sift_keypoints.jpg',img)



