import cv2
import numpy as np

# Load images
large_map = cv2.imread('large_map.jpg', cv2.IMREAD_GRAYSCALE)
map_fragment = cv2.imread('map_fragment.jpg', cv2.IMREAD_GRAYSCALE)

# Apply FFT
f1 = np.fft.fft2(large_map)
f2 = np.fft.fft2(map_fragment)
f1_conj = np.conj(f1)

# Cross power spectrum
cross_power_spectrum = (f1 * f2) / np.abs(f1 * f2)
correlation = np.fft.ifft2(cross_power_spectrum)

# Find peak correlation
y, x = np.unravel_index(np.argmax(np.abs(correlation)), correlation.shape)

# Translate point
print("Translation vector:", x, y)
