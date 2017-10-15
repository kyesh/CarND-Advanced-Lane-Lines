## **Advanced Lane Finding Project**

---


The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1b]: ./cal_out/test_undist.jpg "Undistorted"
[image1a]: ./camera_cal/calibration1.jpg "Orginal"
[image2]: ./straight_cal.png "Road Transformed"
[image3]: ./straight_binary.jpg "Binary Example"
[image4]: ./perspectiveTransformedStriaght.png "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./testk.png "Output"
[video1]: ./output.avi "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I based my work off of this repository that was shown in lecture. https://github.com/udacity/CarND-Camera-Calibration/blob/master/camera_calibration.ipynb

My work is in [calibrateCamera.py](https://github.com/kyesh/CarND-Advanced-Lane-Lines/blob/master/calibrateCamera.py).

1. Updated the rows and column counts to match the ones for the provided checker patern. 
2. I used to glob to select all the camera calibration images we were provieded. 
3. Convert the image to grayscale using `gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`
4. Used `cv2.findChessboardCorners(gray, (cols,rows), None)` to get locations of chessboard corners
5. Outputed drawn chess boards [here](https://github.com/kyesh/CarND-Advanced-Lane-Lines/tree/master/cal_out)
6. Saved the points to a list
7. Used `cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)` to get the calibartion matrixes
8. Used `cv2.undistort(img, mtx, dist, None, mtx)` to undistor the following image

![orginal][image1a]
![unDistorted][image1b]

9. Save the info to [cal.p](.cal_out/cal.p) with pickle

### Pipeline (single images)

#### 1. Example of a distortion-corrected image.

I do this inside my perspective transform file and will go into more detail of how this was done there.
![alt text][image2]

#### 2. Binary Image Production

I didn't think the gradiant thresholding was very useful so I used predominatly image saturation for my thresholding. My work can be found in [createBinaryImage.py](https://github.com/kyesh/CarND-Advanced-Lane-Lines/blob/master/createBinaryImage.py)

1. First I started with the code provided in the lecture material.
a.) Ordered sub-list
b.) Converts the image to HLS colorspace
c.) stores the s and l chanels
d.) use the soble operater to get the gradaint across the image
e.) filters gradiant and satruation channels by provieded inputs
2. After that I created sliders using the OpenCV highgui to test various values of satruation and gradian
3. I determined that gradaiant seemed to add more noise than data so I chose to set the min below the max ignoring that feature

![alt text][image3]

#### 3. Perspective Transform

The code for my perspective Transform is located in [perspectiveTransform.py](https://github.com/kyesh/CarND-Advanced-Lane-Lines/blob/master/perspectivTransform.py)

1. First I need to figure out how to undistort the input image
a.) read in the calibration matrix from the saved pickle file
b.) use `cv2.undistort(img, mtx, dist, None, mtx)`
2. I used gimp on one of the straight road segments to pick out good points for the perspective transform
3. I decided I wanted the overhead view of the lane to have a pixel width of 100
4. I selected the end values for the lanes acordingnly.
5. I then tweaked the height of the overhead image to get a dashed lane lenght of 84px
a.) that would give the end image a rough aspect ratio of 1by1 

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575, 460      | 25, 0        | 
| 705, 460      | 125, 0      |
| 1042, 670     | 125, 500      |
| 265, 670      | 25, 500       |

Here is an example of the result on a straight road segment.

![alt text][image4]

#### 4. Fit Lane Lines

My work for this was done in [fitLaneLines.py](https://github.com/kyesh/CarND-Advanced-Lane-Lines/blob/master/fitLaneLines.py) in the slidingWindowFit function.

1. I started with the sliding window function provied in the lecture material and then made some modifications
2. First is I had to subselect the histagram to get my chanel of intrest using `histogram = histogram[:,2]`
3. Then I added `out_img = binary_warped.copy()` to get a copy of the orginal image to mark up with detections.
4. Orignially I used the histogram to find the lane start positions but opped to used fixed starting value as when little to no know points were avalibe it gave unusual behavior
5. I changed the margin, minpix and numwindows values a bit but I think their effect was negligable
6. I added a conditional statement to only compute a new polyfit if there were sufficent points avalibe. This siginificantly cutdown on the number of weird lines generated when there were only a few points. Example below:
```
if len(leftx) < 200:

		left_fit = []

else:

		left_fit = np.polyfit(lefty, leftx, 2)
```



#### 5. Radius and Position Calculation

My work for this was done in [fitLaneLines.py](https://github.com/kyesh/CarND-Advanced-Lane-Lines/blob/master/fitLaneLines.py) in the computeRadiusAndLanePos function.

1. I used the equations provided in the lecture material for calculating the radius in pixels
```
y_eval = np.max(ploty)
left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
```
2. For lane position I got the average of the two lane start positions and differenced it from the center of the image to figure how far offset the car was from the center of the lane. Negitive corrisponds to left and positive to right.
```
pos = 75 - (left_fitx + right_fitx)/2
```
3. Computing the distance in feet was fairly straight forward as I made my image have a 1by1 aspect ratio using the info that "lane width of 12 feet or 3.7 meters, and the dashed lane lines are 10 feet or 3 meters long each" from the lecture material. I made the lane width 100px in the distination transform so the conversion is multiply by (12/100). The resulting code I used is.
```
text = "left_radius: " + str(lr*12/100) + "ft right_raduis:" + str(rr*12/100) + "ft lane_position:" +str(p*12/100) +"ft"	
cv2.putText(result, text, (50,50), 1, 1, (0,0,255), 1)
 ```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
My work for this was done in [fitLaneLines.py](https://github.com/kyesh/CarND-Advanced-Lane-Lines/blob/master/fitLaneLines.py) in the makePrettyLane function.

1. I create and empty version of the warped image with `color_warp = np.zeros(warped.shape, dtype=np.uint8)`
2. I then pair the x and y points that outline the lane
3. I use `cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))` to draw the lane
4. Then I use `newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))` to warp it into the orginal image shape
5. Finally `result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)` is used to super impose the lane on the orginal image.

![alt text][image6]

---

### Pipeline (video)

#### 1. The End Product!

Here's a [link to my video result](./output.avi)
[video1]

---

### Discussion

I sturggled getting the warpPerspectice working as I orginaly thoughts were supposed to be in (row,col) for not (col,row). I also struggled to get openCV to read videos inside the Conda enviorment. Ultimatly I ran this outside the Conda enviorment. My currently algorithm would not work well if there was a lane change or sharp turns. The narrow slicing I did helps keep cars out of the image but also does not allow you to see the lane fully in sharp turns. Because starting points for the lanes were manually selected they would not generalize well to lane changes when lanes may be in the center of field of view. Improvments could be making it so lanes can be found regardless of starting position so that lane changes can be hanlded. Also better filtering out of cars would allow for wider camera view to detect turns and additional lanes.
