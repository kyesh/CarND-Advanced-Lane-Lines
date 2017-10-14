import perspectivTransform
import createBinaryImage
import numpy as np
import cv2
import matplotlib.pyplot as plt

def slidingWindowFit(binary_warped):
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[round(binary_warped.shape[0]/2):,:], axis=0)
	histogram = histogram[:,2]
	# Create an output image to draw on and  visualize the result
	#out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	#out_img = np.dstack((binary_warped[:,2], binary_warped[:,2], binary_warped[:,2]))*255
	out_img = binary_warped.copy()
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	
	midpoint = np.int(histogram.shape[0]/2)#think this should be shape 1 not 0 to get number of collumns?
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint
	
	#print('left',leftx_base,'right',rightx_base)
	
	# Choose the number of sliding windows
	nwindows = 5 
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 20
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []
	
	# Step through the windows one by one
	for window in range(nwindows):
	    # Identify window boundaries in x and y (and right and left)
	    win_y_low = binary_warped.shape[0] - (window+1)*window_height
	    win_y_high = binary_warped.shape[0] - window*window_height
	    win_xleft_low = leftx_current - margin
	    win_xleft_high = leftx_current + margin
	    win_xright_low = rightx_current - margin
	    win_xright_high = rightx_current + margin
	    # Draw the windows on the visualization image
	    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
	    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
	    # Identify the nonzero pixels in x and y within the window
	    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
	    (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
	    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
	    (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
	    # Append these indices to the lists
	    left_lane_inds.append(good_left_inds)
	    right_lane_inds.append(good_right_inds)
	    # If you found > minpix pixels, recenter next window on their mean position
	    if len(good_left_inds) > minpix:
	        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
	    if len(good_right_inds) > minpix:        
	        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
	    #print('left',leftx_base,'right',rightx_base)
	
	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)
	
	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 
	
	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 255, 0]


	cv2.imwrite('laneHiglights.png',out_img)
	
	return [left_fit, right_fit]
	
def computeRadiusAndLanePos(left_fit, right_fit):
	ploty = np.linspace(0, 499, num=500)# to cover same y-range as image
	
	y_eval = np.max(ploty)
	left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
	right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
	left_fitx = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
	right_fitx = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
	
	pos = 75 - (left_fitx + right_fitx)/2
	return [left_curverad, right_curverad, pos]
	
	
	
	
	
	
	


if __name__ == "__main__":
	import matplotlib.pyplot as plt 
	#img = cv2.imread('test_images/straight_lines1.jpg')
	img = cv2.imread('test_images/test1.jpg')
	warped = perspectivTransform.perspectiveTransform(createBinaryImage.pipeline(img))
	
	l , r = slidingWindowFit(warped)

	lr , rr , p = computeRadiusAndLanePos(l,r)
	
	print('left_radius:',lr*12/100,'ft right_raduis:',rr*12/100,'ft lane_position:',p*12/100,'ft')
	
	histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
	plt.plot(histogram)
	plt.savefig('fig.png')
	
	
