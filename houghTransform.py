import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('road.jpg',0)
original = np.copy(img)
img_p = np.copy(img)
img_c = np.copy(img)
edges = cv2.Canny(img,100,200)

lines = cv2.HoughLines(edges,1,np.pi/180,200)

for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# cv2.imwrite('houghlines3.jpg',img)

# Probabilistic Hough Transform
minLineLength = 200
maxLineGap = 5
lines_p = cv2.HoughLinesP(edges,1,np.pi/180,200,minLineLength,maxLineGap)
for x1_p,y1_p,x2_p,y2_p in lines_p[0]:
    cv2.line(img_p,(x1_p,y1_p),(x2_p,y2_p),(0,255,0),2)

plt.figure(0)
plt.subplot(141),plt.imshow(original,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(img,cmap = 'gray')
plt.title('Hough Transform'), plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(img_p,cmap = 'gray')
plt.title('Probabilistic Hough Transform'), plt.xticks([]), plt.yticks([])

circles = cv2.HoughCircles(img_c,cv2.cv.CV_HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=20)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img_c,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img_c,(i[0],i[1]),2,(0,0,255),3)

plt.subplot(144),plt.imshow(img_c,cmap = 'gray')
plt.title('Hough Transform Circle'), plt.xticks([]), plt.yticks([])


plt.show()