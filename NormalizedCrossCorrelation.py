import cv2
import numpy as np

# Load images (assuming 'large_map' and 'map_fragment' are the image variables)
large_map = cv2.imread('large_map.jpg', cv2.IMREAD_GRAYSCALE)
map_fragment = cv2.imread('map_fragment.jpg', cv2.IMREAD_GRAYSCALE)

# Apply template matching
result = cv2.matchTemplate(large_map, map_fragment, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Determine the top-left corner of the matching area
top_left = max_loc
bottom_right = (top_left[0] + map_fragment.shape[1], top_left[1] + map_fragment.shape[0])

# Draw a rectangle around the matched region
cv2.rectangle(large_map, top_left, bottom_right, 255, 2)

# Display the result
cv2.imshow('Matching Result', large_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
