import cv2
import numpy as np

# Load images
large_map = cv2.imread('large_map.jpg', cv2.IMREAD_GRAYSCALE)
map_fragment = cv2.imread('map_fragment.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(map_fragment, None)
kp2, des2 = sift.detectAndCompute(large_map, None)

# FLANN parameters and matching
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]

# Ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=cv2.DrawMatchesFlags_DEFAULT)

img3 = cv2.drawMatchesKnn(map_fragment, kp1, large_map, kp2, matches, None, **draw_params)

# Display the result
cv2.imshow('Feature Matching with RANSAC', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
