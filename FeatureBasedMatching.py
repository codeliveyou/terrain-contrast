import cv2

# Load images
large_map = cv2.imread('./Images/map_gray_image.jpg', cv2.IMREAD_GRAYSCALE)
map_fragment = cv2.imread('map_fragment.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors
kp1, des1 = orb.detectAndCompute(map_fragment, None)
kp2, des2 = orb.detectAndCompute(large_map, None)

# Create BFMatcher object and match descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw the first 10 matches
img3 = cv2.drawMatches(map_fragment, kp1, large_map, kp2, matches[:10], None, flags=2)

# Display the result
cv2.imshow('Feature Matching', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
