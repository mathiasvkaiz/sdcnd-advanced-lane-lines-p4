import numpy as np
import cv2

class Tracker():

	def __init__(self, window_width, window_height, margin, ym=1, xm=1, smooth_factor=15):
		'''when starting a new instance please be sure to specify all unassigne variables'''
		
		# list that stores all the past (left, right) center set values used for smoothing the output
		self.recent_centers = []

		# the window pixel width of the center values, used to count pixels inside center windows to determine curve values
		self.window_width = window_width

		# the window pixel height of the center values, used to count pixels inside center windows to determine curve values
		# breaks the image into vertcial levels
		self.window_height = window_height

		# The pixel distance in both dirctions to slide (left_window + right_window) template for searching
		self.margin = margin

		# meters per pixel in vertical axis
		self.ym_per_pix = ym

		# meters per pixel in horizontal axis
		self.xm_per_pix = xm

		# smoothing factor
		self.smooth_factor = smooth_factor

	def find_window_centroids(self, warped):
		'''the main tracking function for finding and storing lane segment positions'''

		window_width = self.window_width
		window_height = self.window_height
		margin = self.margin

		# store the (left, right) window centroid positions per level
		window_centroids = []

		# create window template for convolutions
		window = np.ones(window_width)

		# first find the two starting positions for the left and the right lane to get vertical image slice
		# and then convolve the vertical image slice with the window template

		# sum quarter bottom of image to get slice
		# the image is broken up in 9 vertical slices
		# !!! Q&A min 34:50
		l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0) # left half of the image
		l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2 
		r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0) # right half of the image
		r_center = np.argmax(np.convolve(window, r_sum)) - window_width/2 + int(warped.shape[1] / 2) 

		# Add what we found for the first layer
		window_centroids.append((l_center, r_center))

		# go through each layer looking for max pixel locations
		# !!! Q&A min 39:30
		for level in range(1, (int)(warped.shape[0] / window_height)):
			# convolve the window into vertical slice of the image
			image_layer = np.sum(warped[int(warped.shape[0] - (level+1) * window_height):int(warped.shape[0] - level*window_height),:], axis=0)
			conv_signal = np.convolve(window, image_layer)

			# find the best left centroid by using past left center as reference
			# use window width/2 as offset because convolution signal reference is at right side of window
			offset = window_width / 2
			l_min_index = int(max(l_center + offset - margin, 0))
			l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
			l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset # max pixel density per local region
			# find the best left centroid by using past right center as reference
			r_min_index = int(max(r_center + offset - margin, 0))
			r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
			r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset # max pixel density per local region

			# add to centroids
			window_centroids.append((l_center, r_center))

		# append to revent centers
		self.recent_centers.append(window_centroids)

		# return average values of the line centers, avoids jumping / smooth_factor is past n values
		return np.average(self.recent_centers[-self.smooth_factor:], axis=0)

