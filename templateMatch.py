import matplotlib.pyplot as plt
import numpy as np
import cv2

im = cv2.imread('image.png')
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
template = gray[350:450,400:500]
width,height=template.shape

# apply template matching, normed correlated
output=cv2.matchTemplate(gray,template,cv2.TM_CCORR_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(output)

top_left=max_loc
bottom_right = (top_left[0] + width, top_left[1] + height)

cv2.rectangle(gray,top_left, bottom_right, 255, 2)

# plot results

plt.subplot(121)
plt.imshow(output,cmap = 'gray')
plt.title('Matching Result')
plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(gray,cmap = 'gray')
plt.title('Detected Point')
plt.suptitle(top_left)
plt.xticks([]), plt.yticks([])
plt.show()

