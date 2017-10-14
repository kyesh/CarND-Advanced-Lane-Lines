import perspectivTransform
import createBinaryImage
import fitLaneLines
import cv2
import numpy as np

capture = cv2.VideoCapture();
print(capture.open('project_video.mp4'))
#w=int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH ))
#h=int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT ))

retval,img = capture.read()
print(retval)
print(img)
#fourcc = cv2.cv.CV_FOURCC(*'XVID')  # cv2.VideoWriter_fourcc() does not exist
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video_writer = cv2.VideoWriter("output.avi", fourcc, 25, (img.shape[1], img.shape[0]))
video_writerB = cv2.VideoWriter("Binary.avi", fourcc, 25, (img.shape[1], img.shape[0]))
video_writerL = cv2.VideoWriter("LaneTrace.avi", fourcc, 25, (150, 500))
video_writerG = cv2.VideoWriter("WarpedLane.avi", fourcc, 25, (img.shape[1], img.shape[0]))
video_writerGs = cv2.VideoWriter("LaneBox.avi", fourcc, 25, (150, 500))





print(video_writer.isOpened)
i = 0

while(retval == True):
	
	cv2.imwrite('lastImg.png',img)
	
	Bi_img = createBinaryImage.pipeline(img)
	warped, Minv, undist = perspectivTransform.perspectiveTransform(img)
	warped, Minv, n_undist = perspectivTransform.perspectiveTransform(Bi_img)
	
	left_fit , right_fit, laneBox = fitLaneLines.slidingWindowFit(warped)
	print(left_fit,right_fit,len(left_fit),len(right_fit))
	if not len(left_fit)==3:
		left_fit = llf
	if not len(right_fit)==3:
		right_fit = lrf
	llf = left_fit
	lrf = right_fit
	
	lr , rr , p = fitLaneLines.computeRadiusAndLanePos(left_fit,right_fit)
	
	#print('left_radius:',lr*12/100,'ft right_raduis:',rr*12/100,'ft lane_position:',p*12/100,'ft')
	ploty = np.linspace(0, 499, num=500)
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	
	#print(ploty)
	
	newwarp,gs = fitLaneLines.makePrettyLane(warped,left_fitx,right_fitx,ploty,Minv,img)	
	
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
	text = "left_radius: " + str(lr*12/100) + "ft right_raduis:" + str(rr*12/100) + "ft lane_position:" +str(p*12/100) +"ft"	
	cv2.putText(result, text, (50,50), 1, 1, (0,0,255), 1)
	
	
	video_writer.write(result)
	video_writerB.write(cv2.convertScaleAbs(Bi_img))
	video_writerL.write(cv2.convertScaleAbs(laneBox))
	video_writerG.write(newwarp)
	video_writerGs.write(gs)
	i = i + 1
	print(i)
	retval,img = capture.read()
capture.release() 
#video_writer.close()
