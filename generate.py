import numpy as np
import cv2
import glob
import pickle
from moviepy.editor import VideoFileClip
from tracker import Tracker

# Read in the saved obj_points and img_points
dist_pickle = pickle.load(open('camera_cal/calibration_pickle.p', 'rb'))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']

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

# def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
#     sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
#     sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
#     grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
#     scale_factor = np.max(grad_mag)/255 
#     grad_mag = (grad_mag / scale_factor).astype(np.uint8) 
    
#     binary_output = np.zeros_like(grad_mag)
#     binary_output[(grad_mag >= mag_thresh[0]) & (grad_mag <= mag_thresh[1])] = 1
#     return binary_output

# def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
#     sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
#     sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

#     with np.errstate(divide='ignore', invalid='ignore'):
#     	abs_grad_dir = np.absolute(np.arctan(sobel_y / sobel_x))
#     	binary_output =  np.zeros_like(abs_grad_dir)
#     	binary_output[(abs_grad_dir >= thresh[0]) & (abs_grad_dir <= thresh[1])] = 1
    
#     return binary_output

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


def process_image(img):
	
	# undistort the image
	img = cv2.undistort(img, mtx, dist, None, mtx)

	# process image and generate binaries
	grad_x = abs_sobel_thresh(img, orient='x', thresh=(12, 255)) # like canny transform
	grad_y = abs_sobel_thresh(img, orient='y', thresh=(25, 255))
	c_binary = color_thresh(img, s_thresh=(100, 255), v_thresh=(50, 255))
	
	preprocessed = np.zeros_like(img[:, :, 0])
	preprocessed[(grad_x == 1) & (grad_y == 1) | (c_binary == 1)] = 255

	# warp image
	img_size = (img.shape[1], img.shape[0])
	# mask values
	top_left = [580, 450]
	top_right = [720, 450]
	bottom_left = [190, 720]
	bottom_right = [1190, 720]

	proj_top_left = [320, 0]
	proj_top_right = [1000, 0]
	proj_bottom_left = [320, 720]
	proj_bottom_right = [1000, 720]
	
	# bottom_width = 0.76 # bottom width percentage
	# middle_width = 0.08 # middle height percentage
	# height_percentage = 0.62 # height percentage
	# bottom_trim = 0.935 # from top to bottom percentage
	# offset = img_size[0] * 0.25
	# dst = np.float32([[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]], [offset, img_size[1]]])

	src = np.float32([bottom_left, top_left, top_right, bottom_right])
	dst = np.float32([proj_bottom_left, proj_top_left, proj_top_right, proj_bottom_right])

	# w, h = 1280, 720
	# x, y = 0.5*w, 0.8*h
	# src = np.float32([[200./1280*w,720./720*h], [453./1280*w,547./720*h], [835./1280*w,547./720*h], [1100./1280*w,720./720*h]])
	# dst = np.float32([[(w-x)/2.,h], [(w-x)/2.,0.82*h], [(w+x)/2.,0.82*h], [(w+x)/2.,h]])

	# perform perspective transform
	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	warped = cv2.warpPerspective(preprocessed, M, img_size, flags=cv2.INTER_LINEAR)

	# tracking
	window_width = 25
	window_height = 80
	tracker = Tracker(window_width=window_width, window_height=window_height, margin=25, ym=10/720, xm=4/384, smooth_factor=15)
	window_centroids = tracker.find_window_centroids(warped)



	# windows
	# points used to draw left and right windows
	l_points = np.zeros_like(warped)
	r_points = np.zeros_like(warped)

	left_x = []
	right_x = []


	# go through each level and calc windows
	for level in range(0, len(window_centroids)):
		# add center value found in frame to the list of lane points per left, right
		left_x.append(window_centroids[level][0])
		right_x.append(window_centroids[level][1])

		# window_mask is a function to draw window areas
		l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
		r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)

		# add graphic points from windo mask to total pixels
		l_points[(l_points == 255) | (l_mask == 1)] = 255
		r_points[(r_points == 255) | (r_mask == 1)] = 255

	
	# # draw windows
	# template = np.array(r_points + l_points, np.uint8) # add lft and right window pixels together
	# zero_channel = np.zeros_like(template) # zero color channel
	# template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8) # green the window pixels
	# warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8) # the original road pixels in 3 channels
	# result = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the original road image with the window results





	# lane curves
	# fit the lane boundaries to the left, right and center positions found
	y_vals = range(0, warped.shape[0]) # used fo conitnous space in resolution with one pixel
	res_y_vals = np.arange(warped.shape[0] - (window_height/2), 0, -window_height) # fitting to box centers

	left_fit = np.polyfit(res_y_vals, left_x, 2) # fitting to box centers
	left_fitx = left_fit[0]*y_vals*y_vals + left_fit[1]*y_vals + left_fit[2] # find coefficients of that curve
	left_fitx = np.array(left_fitx, np.int32)

	right_fit = np.polyfit(res_y_vals, right_x, 2) # fitting to box centers
	right_fitx = right_fit[0]*y_vals*y_vals + right_fit[1]*y_vals + right_fit[2] # find coefficients of that curve
	right_fitx = np.array(right_fitx, np.int32)

	# create arrays for lines
	left_lane = np.array(list(zip(np.concatenate((left_fitx - window_width/2, left_fitx[::-1] + window_width/2), axis=0), np.concatenate((y_vals, y_vals[::-1]), axis=0))), np.int32)
	right_lane = np.array(list(zip(np.concatenate((right_fitx - window_width/2, right_fitx[::-1] + window_width/2), axis=0), np.concatenate((y_vals, y_vals[::-1]), axis=0))), np.int32)
	inner_lane = np.array(list(zip(np.concatenate((left_fitx - window_width/2, right_fitx[::-1] + window_width/2), axis=0), np.concatenate((y_vals, y_vals[::-1]), axis=0))), np.int32)

	road = np.zeros_like(img)
	# road_bg = np.zeros_like(img)
	cv2.fillPoly(road, [left_lane], color=[255, 0, 0])
	cv2.fillPoly(road, [right_lane], color=[0, 0, 255])
	cv2.fillPoly(road, [inner_lane], color=[0, 255, 0])
	# cv2.fillPoly(road_bg, [left_lane], color=[255, 255, 255])
	# cv2.fillPoly(road_bg, [right_lane], color=[255, 255, 255])

	road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)
	# road_warped_bg = cv2.warpPerspective(road_bg, Minv, img_size, flags=cv2.INTER_LINEAR)

	# base = cv2.addWeighted(img, 1.0, road_warped_bg, -1.0, 0.0)
	result = cv2.addWeighted(img, 1.0, road_warped, 1.0, 0.0)





	# calculate offset
	ym_per_pix = tracker.ym_per_pix
	xm_per_pix = tracker.xm_per_pix

	# radius of curvature
	curve_fit_cr = np.polyfit(np.array(res_y_vals, np.float32) * ym_per_pix, np.array(left_x, np.float32) * xm_per_pix, 2)
	curve_rad = ((1 + (2 * curve_fit_cr[0] * y_vals[-1] * ym_per_pix + curve_fit_cr[1])**2)**1.5) / np.absolute(2 * curve_fit_cr[0])

	camera_center = (left_fitx[-1] + right_fitx[-1]) / 2. # -1 to have the closest to the car
	center_diff = (camera_center - warped.shape[1] / 2) * xm_per_pix
	side_pos = 'left'
	if center_diff <= 0:
		side_pos = 'right'

	# draw the text shwoing curvature, offset
	cv2.putText(result, 'Radius of Curvature = ' + str(round(curve_rad, 3)) + '(m)', (50, 50), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
	cv2.putText(result, 'Vehicle is ' + str(abs(round(center_diff, 3))) + 'm ' + side_pos + ' of center', (50, 100), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)


	return result



# Make a list of test images
images_list = glob.glob('./test_images/test*.jpg')

for idx, fname in enumerate(images_list):
	# read the image
	img = cv2.imread(fname)

	result = process_image(img)

	# save the result
	write_name = './test_images/tracked' + str(idx) + '.jpg'
	cv2.imwrite(write_name, result) 


# process video
# input_video = 'project_video.mp4'
# output_video = 'output_video.mp4'

# clip = VideoFileClip(input_video)
# video_clip = clip.fl_image(process_image)
# video_clip.write_videofile(output_video, audio=False)