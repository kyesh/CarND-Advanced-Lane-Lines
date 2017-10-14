import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def perspectiveTransform(img):
	
	# Read in the saved camera matrix and distortion coefficients
	# These are the arrays you calculated using cv2.calibrateCamera()
	dist_pickle = pickle.load( open( "cal_out/cal.p", "rb" ) )
	mtx = dist_pickle["mtx"]
	dist = dist_pickle["dist"]

	undist = cv2.undistort(img, mtx, dist, None, mtx)

	#cv2.imwrite('straight_cal.png',undist)
	#offset = 100
	img_size = (img.shape[1], img.shape[0])

	#src = np.float32([[400,400],[800,400],[800,800],[400,800]])
	#dst = np.float32([[400,400],[800,400],[800,800],[400,800]])

	#src = np.float32([[435,617],[670,1055], [435,661],[670,240]])

	src = np.float32([[575,460], [705,460], [1042,670], [265,670]])

	#dst = np.float32([[offset, offset], [img_size[0]-offset, offset], [img_size[0]-offset, img_size[1]-offset], [offset, img_size[1]-offset]])

	#dst = np.float32([[00,616], [00,662], [670,1055], [670,240]])
	#dst = np.float32([[offset, offset], [img_size[0]-offset, offset], [img_size[0]-offset, img_size[1]-offset], [offset, img_size[1]-offset]])
	#dst = np.float32([[400,400], [400,img.shape[1]-400], [img.shape[0]-400,img.shape[1]-400], [img.shape[0]-400,400]])
	imgheight = 500 
	dst = np.float32([[25,0],[125,0],[125,imgheight],[25,imgheight]])

	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	warped = cv2.warpPerspective(undist, M, (150,imgheight))

	return warped, Minv, undist

if __name__ == "__main__":

	# Read in an image
	img = cv2.imread('test_images/straight_lines1.jpg')

	warped = perspectiveTransform(img)

	cv2.imwrite('test.png',warped)

	#img = cv2.imread('camera_cal/calibration1.jpg')




