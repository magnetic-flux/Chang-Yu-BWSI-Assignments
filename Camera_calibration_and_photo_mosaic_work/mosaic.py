# Result can be found in Mosaic.png

import cv2
import numpy as np
import glob

def load_calibration(calibration_file):
    with np.load(calibration_file) as data:
        camera_matrix = data['array_A']
        dist_coeffs = data['array_B']
    return camera_matrix, dist_coeffs

def undistort_image(image, camera_matrix, dist_coeffs):
    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    dst = cv2.undistort(image, camera_matrix, dist_coeffs, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

def match_features(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return points1, points2

def warp_images(base_image, current_image, h_matrix):
    h1, w1 = base_image.shape[:2]
    h2, w2 = current_image.shape[:2]
    
    # Corners of the base image
    base_corners = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    
    # Corners of the current image after warping
    current_corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(current_corners, h_matrix)
    
    # Combine the corners to find the dimensions of the final mosaic
    all_corners = np.concatenate((base_corners, warped_corners), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    translation_dist = [-x_min, -y_min]
    h_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    
    # Warp the current image
    warped_image = cv2.warpPerspective(current_image, h_translation @ h_matrix, (x_max - x_min, y_max - y_min))
    
    # Place the base image in the final mosaic
    mosaic = warped_image.copy()
    mosaic[translation_dist[1]:translation_dist[1] + h1, translation_dist[0]:translation_dist[0] + w1] = base_image
    
    return mosaic

def create_mosaic(images, camera_matrix, dist_coeffs):
    undistorted_images = [undistort_image(image, camera_matrix, dist_coeffs) for image in images]
    base_image = undistorted_images[0]

    mosaic = base_image
    for image_num in range(1, len(undistorted_images)):
        current_image = undistorted_images[image_num]
        pts1, pts2 = match_features(mosaic, current_image)
        h_matrix, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        mosaic = warp_images(mosaic, current_image, h_matrix)
    
    return mosaic

calibration_file = '/home/magnetic/laboratory_2024/week_1_Hw/camera_calibration_photo_mosaic/camera_data.npz'
camera_matrix, dist_coeffs = load_calibration(calibration_file)

images = [cv2.imread(file) for file in glob.glob('/home/magnetic/laboratory_2024/week_1_Hw/camera_calibration_photo_mosaic/Latin Student Center/*.jpg')]

mosaic_image = create_mosaic(images, camera_matrix, dist_coeffs)

cv2.imwrite('mosaic.jpg', mosaic_image)
cv2.imshow('Mosaic', mosaic_image)
cv2.waitKey(0)
cv2.destroyAllWindows()