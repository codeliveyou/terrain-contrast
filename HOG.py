import cv2
import numpy as np
from skimage.feature import hog
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

def show_image(image, title="Image"):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()

def calculate_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    # Note that channel_axis is removed because the image should be single-channel (grayscale)
    fd, hog_image = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                        cells_per_block=cells_per_block, visualize=True)
    return fd


def compare_hog_features(imageA, imageB):
    # Convert images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Resize images to ensure scale invariance
    grayA = cv2.resize(grayA, (128, 128))
    grayB = cv2.resize(grayB, (128, 128))

    # Calculate HOG features
    hogA = calculate_hog_features(grayA)
    hogB = calculate_hog_features(grayB)

    # Compute cosine similarity between HOG features
    similarity = 1 - cosine(hogA, hogB)
    return similarity

# Load images
imageA = cv2.imread('A.jpg')
imageB = cv2.imread('D.jpg')

# Compare images using HOG
similarity_score = compare_hog_features(imageA, imageB)
print(f"HOG Similarity Score: {similarity_score}")
