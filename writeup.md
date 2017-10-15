## Writeup Template

---

**Advanced Lane Finding Project**

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
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

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



![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
