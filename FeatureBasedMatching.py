import cv2

# Load images
large_map = cv2.imread('./Images/map_gray_image.jpg', cv2.IMREAD_GRAYSCALE)
map_fragment = cv2.imread('Images/t_gray_image.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=5000)

# Detect keypoints
kp1 = orb.detect(map_fragment, None)
kp2 = orb.detect(large_map, None)

# Draw keypoints
keypoint_img1 = cv2.drawKeypoints(map_fragment, kp1, None, color=(255, 0, 0), flags=0)
keypoint_img2 = cv2.drawKeypoints(large_map, kp2, None, color=(255, 0, 0), flags=0)

# Display keypoints
cv2.imshow('Keypoints 1', keypoint_img1)
cv2.imshow('Keypoints 2', keypoint_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
