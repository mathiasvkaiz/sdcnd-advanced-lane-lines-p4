## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./output_images/corners_found1.jpg "Camera Calibration"
[image2]: ./output_images/undistorted0.jpg "Undistorted"
[image3]: ./output_images/preprocessed0.jpg "Binary"
[image4]: ./output_images/masked0.jpg "Region of Interest"
[image5]: ./output_images/warped0.jpg "Warped"
[image6]: ./output_images/windowed0.jpg "Windows"
[image7]: ./examples/curvature.png "Curvature Formula"
[image8]: ./output_images/result0.jpg "Final Result"

[video1]: ./output_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it and here is a link to my [project code](https://github.com/mathiasvkaiz/sdcnd-advanced-lane-lines-p4/blob/master/Advanced_Lane_Lines.ipynb)

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the fourth code cell of the IPython notebook (see reference above) with headline `Camera Calibration`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `obj_p` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

All relevant functions for creting binary image are placed in the sixth code cell. The pipline uses `abs_sobel_thresh`, `color_thresh` both called in function `preprocess`. After that a region of interest is defined to subtract unnecessary interferences by `region_of_interest`.

I used a combination of absolute Sobel on x and y axis combined with color thresholds to generate a binary image.

![alt text][image3]

After that i applied a region of interest to filter out all unnecessary stuff. So the focus lies on the trapezoid ahead on the road and not all the pixels outside this mask.

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears sixth code cell of the IPython notebook.  The `warp()` function takes as inputs an image (`img`).  I chose the global (defined in first code cell) hardcode the source and destination points in the following manner:

```python
top_left = [580, 450]
top_right = [720, 450]
bottom_left = [190, 720]
bottom_right = [1190, 720]

proj_top_left = [320, 0]
proj_top_right = [1000, 0]
proj_bottom_left = [320, 720]
proj_bottom_right = [1000, 720]

src = np.float32([bottom_left, top_left, top_right, bottom_right])
dst = np.float32([proj_bottom_left, proj_top_left, proj_top_right, proj_bottom_right])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 450      | 320, 0        | 
| 720, 450      | 1000, 0       |
| 190, 720      | 320, 720      |
| 1190, 720     | 1000, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In function `window_slide` in cell six of the notebook i apply the polynomial wfit and sliding windows.
Most functionality is commented in the code but i want to give abrief summary:

After having the lines clearly visible in the warped image we need to define wich pixels are part of the line and which are not. Here comes a histogram into place. With that historgram we can identify regions(columns) of the image where we have non zero pixels by showing peaks in the histogram. Those peaks are used as starting points and form those points on we draw asmall rectangles/windows along the polynomial line.

![alt text][image6]

Based on the given windows we can now define (including some mrgin) a thicker line. This is our identified lane.
after that the function `projection` in code cell six projects this lane back onto the original undistorted image using inverse matrix perspective transformation.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

You can find this code parts in cell six in function `finalize`. Based on the global variabls for transforming a pixel onto meters.

```python
conv_y = 30/720 # meters per pixel in y dimension
conv_x = 3.7/700 # meters per pixel in x dimension
```

The curvature is calulated based on the found pixels for left and right lane with respect to the given y pixels / height for each x pixel.

The following formula is used applying also the conversion factors for pixel to meters.

![alt text][image7]


The position is calculated by getting the closest inline pixels from the found pixels (left/right) of the lane and add them together. After that we minus this result with the half of the total image width and apply the conversion factor.



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the code parts in cell six in function `finalize`.  Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach i took is a very basic one. First i used brute force techniques to get good thresholds based on different color spaces and gradienst / absolute Sobel values. After that i tried out the polynomial fit and also the convolution solution for finding the lanes. I decided to use the convolution approach as in my opinion it is more code on one hand but on the other hand much more easier for me to extend this to several new approaches.

We can think of having an average lane calculation of several frames before the actual one to avoid flipping around. Also we could check if the found lane makes sense in terms of curvature. It would not make much sense to have a curvature completely opposite in the next frame discarding this one and taking the average of several frames. So this could also lead to a much smoother calculation of the lanes. There are many more apporaches than used in the notebook and for getting good results in the challenges these approaches should be applied. 

What could also lad to some noise is the switch between sunlight and shadows combined with different road patterns. This could lead to noise as well. We could avoid this by using some flexible color space calculation based on luminosity. 
