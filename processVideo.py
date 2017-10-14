import perspectivTransform
import createBinaryImage
import fitLaneLines
import cv2

capture = cv2.VideoCapture();
print(capture.open('project_video.mp4'))
#w=int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH ))
#h=int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT ))

retval,img = capture.read()
print(retval)
print(img)
#fourcc = cv2.cv.CV_FOURCC(*'XVID')  # cv2.VideoWriter_fourcc() does not exist
video_writer = cv2.VideoWriter("output.avi", -1, 30, (img.shape[1], img.shape[0]))

i = 0

while(retval == True):
	
	Bi_img = createBinaryImage.pipeline(img)
	warped, Minv, undist = perspectivTransform.perspectiveTransform(img)
	warped, Minv, n_undist = perspectivTransform.perspectiveTransform(Bi_img)
	
	left_fit , right_fit = slidingWindowFit(warped)
	
	
	
	lr , rr , p = computeRadiusAndLanePos(left_fit,right_fit)
	
	print('left_radius:',lr*12/100,'ft right_raduis:',rr*12/100,'ft lane_position:',p*12/100,'ft')
	ploty = np.linspace(0, 499, num=500)
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	
	newwarp = makePrettyLane(warped,left_fitx,right_fitx,Minv,img)	
	
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
	
	

	
	video_writer.write(result)
	i = i + 1
	print(i)
	retval,img = capture.read()
capture.release() 
#video_writer.close()
