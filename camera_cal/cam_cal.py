import numpy as np
import cv2
import glob
import pickle

dimensions = (9, 6)
obj_points = []
img_points = []

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
obj_p = np.zeros((6*9, 3), np.float32)
obj_p[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Make a list of images to calibrate
images_list = glob.glob('./calibration*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images_list):
    # Read each image
    img = cv2.imread(fname)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, dimensions, None)

    # If corners are found, add object points, image points and image to images array
    if ret == True:
        print('working on ', fname)
        obj_points.append(obj_p)
        img_points.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, dimensions, corners, ret)
        write_name = 'corners_found' + str(idx) + '.jpg'
        cv2.imwrite(write_name, img)

# load image for reference
img = cv2.imread('./calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# Camra calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

# Save the camra calibration result for later use
print('save pickle')
dist_pickle = {}
dist_pickle['mtx'] = mtx
dist_pickle['dist'] = dist
pickle.dump(dist_pickle, open('./calibration_pickle.p', 'wb'))