import cv2
import numpy as np

# Load images correctly by specifying the correct path
# Replace 'large_map.jpg' and 'map_fragment.jpg' with the correct paths if needed
large_map_path = './Images/map_gray_image.jpg'
map_fragment_path = 'Images/t_gray_image.jpg'

# Check if the file exists to avoid loading errors
import os
if not os.path.exists(large_map_path) or not os.path.exists(map_fragment_path):
    raise FileNotFoundError("One or more image files could not be found.")

large_map = cv2.imread(large_map_path, cv2.IMREAD_GRAYSCALE)
map_fragment = cv2.imread(map_fragment_path, cv2.IMREAD_GRAYSCALE)

# Check if images are loaded properly
if large_map is None or map_fragment is None:
    raise Exception("Failed to load images. Check file paths and image file integrity.")

# Create a mask for the map fragment where erased parts are set to 255
# We need to make sure mask and map_fragment have the same dimensions
mask = np.where(map_fragment == 255, 0, 255).astype(np.uint8)

# Apply template matching using the mask
result = cv2.matchTemplate(large_map, map_fragment, cv2.TM_CCOEFF_NORMED, mask=mask)
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
