import matplotlib.pyplot as plt
import numpy as np
import cv2

def match_strips(strip_left,strip_right,b):
	num_blocks=strip_left.shape[1]/b
	disparity = np.zeros([1,num_blocks])
	for block in np.arange(0,num_blocks):
		x_left=block*b
		patch_left=strip_left[:,x_left:(x_left+b)]
		x_right=find_best_match(patch_left,strip_right)
		disparity[0,block]=x_left-x_right
	print disparity, num_blocks
	return disparity, num_blocks

def find_best_match(patch, strip):
	min_diff=np.inf
	best_x=None
	for x in np.arange(0,(strip.shape[1]-patch.shape[1]-1)):
		temp_patch = strip[:,x:(x+patch.shape[1])]
		diff=np.sum((patch[:]-temp_patch[:])**2)
		if diff <min_diff:
			min_diff=diff
			best_x=x

	return best_x

left = cv2.imread('left.png')
right = cv2.imread('right.png')

# convert to gray and normalize for easier computation
left_gray = cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)/255.0
right_gray = cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)/255.0

patch_loc=[96, 136]
patch_size=[80, 80]

# patch from left image
patch_left=left_gray[(patch_loc[0]-1):(patch_loc[0]+patch_size[0]-1),(patch_loc[1]-1):(patch_loc[1]+patch_size[1]-1)]
# strip from right image
strip_right=right_gray[(patch_loc[0]-1):(patch_loc[0]+patch_size[0]-1),:]
strip_left=left_gray[(patch_loc[0]-1):(patch_loc[0]+patch_size[0]-1),:]

best_x=find_best_match(patch_left,strip_right)
patch_right=right_gray[(patch_loc[0]-1):(patch_loc[0]+patch_size[0]-1),best_x:(best_x+patch_size[1])]
disparity,num_blocks=match_strips(strip_left,strip_right,patch_size[0])

# plot results
plt.figure(0)
plt.subplot(121)
plt.imshow(left,cmap = 'gray')
plt.title('left')
plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(right,cmap = 'gray')
plt.title('right')
plt.xticks([]), plt.yticks([])


plt.figure(1)
plt.subplot(311)
plt.imshow(patch_left,cmap = 'gray')
plt.title('patch_left')
plt.xticks([]), plt.yticks([])
plt.subplot(312)
plt.imshow(strip_right,cmap = 'gray')
plt.title('strip_right')
plt.xticks([]), plt.yticks([])
plt.subplot(313)
plt.imshow(patch_right,cmap = 'gray')
plt.title('patch_right')
plt.suptitle(best_x)
plt.xticks([]), plt.yticks([])

# compare results with template match
new_left_gray=cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)
template = new_left_gray[(patch_loc[0]-1):(patch_loc[0]+patch_size[0]-1),(patch_loc[1]-1):(patch_loc[1]+patch_size[1]-1)]
width,height=template.shape
new_strip_right=cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)
# new_strip_right=new_strip_right[(patch_loc[0]-1):(patch_loc[0]+patch_size[0]-1),:]

# apply template matching, normed correlated
output=cv2.matchTemplate(new_strip_right,template,cv2.TM_CCORR_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(output)

top_left=max_loc
bottom_right = (top_left[0] + width, top_left[1] + height)

cv2.rectangle(new_strip_right,top_left, bottom_right, 255, 2)


plt.figure(2)
plt.imshow(new_strip_right,cmap = 'gray')
plt.title('template match')
plt.suptitle(top_left)
plt.xticks([]), plt.yticks([])

plt.figure(3)

x=np.arange(0,num_blocks)
plt.plot(x,disparity[0,:],'r-')
plt.title('disparity')
plt.suptitle('Match strip')

plt.show()