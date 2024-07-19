import numpy as np
import cv2
import glob

def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse

checkerboard_dims = (8, 6)
objp = np.zeros((checkerboard_dims[0]*checkerboard_dims[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:checkerboard_dims[0],0:checkerboard_dims[1]].T.reshape(-1,2)

objpoints = []
imgpoints = []

img = glob.glob('/home/magnetic/laboratory_2024/week_1_Hw/camera_calibration_photo_mosaic/calibration_photos/*.jpg')

gray_list = []

for image in img:
    read_image = cv2.imread(image)
    image_gray = cv2.cvtColor(read_image, cv2.COLOR_BGR2GRAY)
    gray_list.append(image_gray)

    ret, corners = cv2.findChessboardCorners(image_gray, checkerboard_dims, None)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(image_gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        cv2.waitKey(100)
    
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_list[0].shape[::-1], None, None)

print(mtx)

#np.savez("/home/magnetic/laboratory_2024/week_1_Hw/camera_calibration_photo_mosaic/camera_data", array_A=mtx, array_B=dist)

error = 0
for i in range(len(objpoints)):
    projected_points, __ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = error + cv2.norm(projected_points, imgpoints[i], cv2.NORM_L2) / len(projected_points)

mean_error = error / len(objpoints)
print("Average error: " + str(mean_error))
