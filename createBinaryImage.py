import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import perspectivTransform


# Edit this function to create your own pipeline.

#Select yellow and white from previous reviewer
def select_yellow(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower = np.array([20,60,60])
	upper = np.array([38,174, 250])
	mask = cv2.inRange(hsv, lower, upper)
	
	return mask

def select_white(image):
	lower = np.array([202,202,202])
	upper = np.array([255,255,255])
	mask = cv2.inRange(image, lower, upper)
	
	return mask

def pipeline(img, s_thresh=(150, 255), sx_thresh=(147, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary

if __name__ == "__main__":

	smax = 255;
	smin = 150;
	sxmax = 100;
	sxmin = 147;

	#image = mpimg.imread('test_images/straight_lines1.jpg')
	#image = cv2.imread('test_images/test1.jpg')
	image = cv2.imread('test_images/straight_lines1.jpg')

	def updateSmax(pos):
		global smax
		smax = pos
		updateImg();
		return

	def updateSmin(pos):
		global smin
		smin = pos
		updateImg();
		return

	def updateSxmax(pos):
		global sxmax
		sxmax=pos
		updateImg();
		return

	def updateSxmin(pos):
		global sxmin
		sxmin = pos
		updateImg();
		return

	def updateImg():
		global image
		result = pipeline(image, s_thresh=(smin, smax), sx_thresh=(sxmin, sxmax))
		warpedresult = perspectivTransform.perspectiveTransform(result)
		#cv2.imshow('img', image)
		cv2.imshow('Thresh Image',result)
		cv2.imshow('Warped Thresh',warpedresult)
		
		return

	cv2.namedWindow('Original Image')
	cv2.namedWindow('Thresh Image')
	cv2.namedWindow('Warped Thresh')
	cv2.namedWindow('Track Bars')

	result = pipeline(image, s_thresh=(smin, smax), sx_thresh=(sxmin, sxmax))
	cv2.imshow('Thresh Image',result)
	cv2.imshow('Original Image',image)

	cv2.createTrackbar('smax', 'Track Bars', smax, 255, updateSmax) 
	cv2.createTrackbar('smin', 'Track Bars', smin, 255, updateSmin)         
	cv2.createTrackbar('sxmax', 'Track Bars', sxmax, 255, updateSxmax)         
	cv2.createTrackbar('sxmin', 'Track Bars', sxmin, 255, updateSxmin)         

	cv2.waitKey(0)

	#result = pipeline(image)

	# Plot the result
	#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
	#f.tight_layout()

	#ax1.imshow(image)
	#ax1.set_title('Original Image', fontsize=40)
	cv2.imwrite('test.png',result)
	#ax2.imshow(result)
	#ax2.set_title('Pipeline Result', fontsize=40)
	#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
