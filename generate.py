import numpy as np
import cv2
import glob
import pickle
import collections
from moviepy.editor import VideoFileClip

# Read in the saved obj_points and img_points
dist_pickle = pickle.load(open('camera_cal/calibration_pickle.p', 'rb'))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']

# mask values
top_left = [580, 450]
top_right = [720, 450]
bottom_left = [190, 720]
bottom_right = [1190, 720]

proj_top_left = [320, 0]
proj_top_right = [1000, 0]
proj_bottom_left = [320, 720]
proj_bottom_right = [1000, 720]

output_images_path = './output_images/'

# The lessons binary creating was a good starting point, but using color thresholding as well 
# we have a lot more possibilities to get better results
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return binary_output


def color_thresh(image, s_thresh=(0, 255), v_thresh=(0, 255)):
	hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	s_channel = hls[:, :, 2]
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1


	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	v_channel = hsv[:, :, 2]
	v_binary = np.zeros_like(v_channel)
	v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

	binary_output = np.zeros_like(s_channel)
	binary_output[(s_binary == 1) & (v_binary == 1)] = 1

	return binary_output

def window_mask(width, height, img, center, level):
	output = np.zeros_like(img)
	output[int(img.shape[0] - (level+1) * height) : int(img.shape[0] - level * height), max(0, int(center - width)):min(int(center + width), img.shape[1])] = 1

	return output


def region_of_interest(img, vertices=None):
    '''
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    '''
    
    if vertices == None:
        vertices = np.array([[bottom_left, (top_left[0], top_left[1]), (top_right[0], top_right[1]), bottom_right]], dtype=np.int32)

    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def process_image(img, save_ouput_file=False, is_first_frame=True):
	
	# undistort the image
	img = cv2.undistort(img, mtx, dist, None, mtx)
	img_size = (img.shape[1], img.shape[0])
	
	# save the undistorted
	if save_ouput_file==True:
		write_name = output_images_path + 'undistorted' + str(idx) + '.jpg'
		cv2.imwrite(write_name, img) 




	# process image and generate binaries
	grad_x = abs_sobel_thresh(img, orient='x', thresh=(12, 255)) # like canny transform
	grad_y = abs_sobel_thresh(img, orient='y', thresh=(25, 255))
	c_binary = color_thresh(img, s_thresh=(100, 255), v_thresh=(50, 255))
	
	preprocessed = np.zeros_like(img[:, :, 0])
	preprocessed[(grad_x == 1) & (grad_y == 1) | (c_binary == 1)] = 255

	# save the preprocessed
	if save_ouput_file==True:
		write_name = output_images_path + 'preprocessed' + str(idx) + '.jpg'
		cv2.imwrite(write_name, preprocessed) 



	# apply region of interest
	masked = region_of_interest(preprocessed)

	# save the masked
	if save_ouput_file==True:
		write_name = output_images_path + 'masked' + str(idx) + '.jpg'
		cv2.imwrite(write_name, masked) 


	# warp image
	src = np.float32([bottom_left, top_left, top_right, bottom_right])
	dst = np.float32([proj_bottom_left, proj_top_left, proj_top_right, proj_bottom_right])

	# perform perspective transform
	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	warped = cv2.warpPerspective(masked, M, img_size, flags=cv2.INTER_LINEAR)

	# save the warped
	if save_ouput_file==True:
		write_name = output_images_path + 'warped' + str(idx) + '.jpg'
		cv2.imwrite(write_name, warped) 


	binary_warped = np.copy(warped)

	if is_first_frame == True:

		is_first_frame = False

		# Assuming you have created a warped binary image called "binary_warped"
		# Take a histogram of the bottom half of the image
		histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
		# Create an output image to draw on and  visualize the result
		windowed_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
		# Find the peak of the left and right halves of the histogram
		# These will be the starting point for the left and right lines
		midpoint = np.int(histogram.shape[0]/2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint

		# Choose the number of sliding windows
		nwindows = 9
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
		margin = 100
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
			cv2.rectangle(windowed_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
			(0,255,0), 2) 
			cv2.rectangle(windowed_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
			(0,255,0), 2) 
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

		# Generate x and y values for plotting
		ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		windowed_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		windowed_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

		# save the windowed
		if save_ouput_file==True:
			write_name = output_images_path + 'windowed' + str(idx) + '.jpg'
			cv2.imwrite(write_name, windowed_img) 

	else:
		# Assume you now have a new warped binary image 
		# from the next frame of video (also called "binary_warped")
		# It's now much easier to find line pixels!
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		margin = 100
		left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
		left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
		left_fit[1]*nonzeroy + left_fit[2] + margin))) 

		right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
		right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
		right_fit[1]*nonzeroy + right_fit[2] + margin)))  

		# Again, extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds] 
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]
		# Fit a second order polynomial to each
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)
		# Generate x and y values for plotting
		ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Create an image to draw the lines on
	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
	# Combine the result with the original image
	result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension
	
	# curvature
	leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
	y_eval = np.max(ploty)
	curve_rad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])

	# offset of center
	camera_center = (left_fitx[-1] + right_fitx[-1]) / 2. # -1 to have the closest to the car
	center_diff = (camera_center - warped.shape[1] / 2) * xm_per_pix

	# # draw the text showing curvature, offset of center
	cv2.putText(result, 'Radius of Curvature: ' + str(round(curve_rad, 3)) + '(m)', (50, 50), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
	cv2.putText(result, 'Vehicle position of center: ' + str(round(center_diff, 3)) + '(m)', (50, 100), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)

	return result



# Make a list of test images
images_list = glob.glob('./test_images/test*.jpg')

for idx, fname in enumerate(images_list):
	# read the image
	img = cv2.imread(fname)

	result = process_image(img, True)

	# save the result
	write_name = output_images_path + 'result' + str(idx) + '.jpg'
	cv2.imwrite(write_name, result) 




# process video
# input_video = 'project_video.mp4'
# output_video = 'output_video.mp4'

# clip = VideoFileClip(input_video)
# video_clip = clip.fl_image(process_image)
# video_clip.write_videofile(output_video, audio=False)