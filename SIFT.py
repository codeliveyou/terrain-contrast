import cv2
import numpy as np

def drawMatches(img1, kp1, img2, kp2, matches):
    # Check the number of channels and convert to BGR if they are grayscale
    if len(img1.shape) == 2:  # Grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:  # Grayscale
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    # Create a blank image with enough space to place both images side by side
    out = np.zeros((max(rows1, rows2), cols1 + cols2, 3), dtype='uint8')
    out[:rows1, :cols1, :] = img1
    out[:rows2, cols1:, :] = img2

    # Draw circles and lines for the matches
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)
        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0), 1)

    return out

def compare(imageA, imageB):
    sift = cv2.SIFT_create()  # Initiate SIFT detector

    kp1, des1 = sift.detectAndCompute(imageA, None)  # Keypoints and descriptors
    kp2, des2 = sift.detectAndCompute(imageB, None)

    bf = cv2.BFMatcher()  # BFMatcher with default params
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    print(matches[-1].distance)

    img3 = drawMatches(imageA, kp1, imageB, kp2, matches[:25])  # Draw first 25 matches
    cv2.imshow('Matched Features', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load images and ensure they are in the correct format
imageA = cv2.imread('./images/D.jpg')
imageB = cv2.imread('./images/A.jpg')

if imageA is None or imageB is None:
    raise ValueError("One or both images did not load. Check the file paths.")
if len(imageA.shape) == 3:  # Convert to grayscale if necessary
    imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
if len(imageB.shape) == 3:
    imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

compare(imageA, imageB)












# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
# from tkinter.filedialog import askopenfilename

# filename1 = askopenfilename(filetypes=[("A","*.jpg")]) # queryImage
# filename2 = askopenfilename(filetypes=[("D","*.jpg")]) # trainImage

# img1=cv2.imread(filename1,4)
# img2=cv2.imread(filename2,4)

# # Initiate SURF detector
# surf=cv2.xfeatures2d.SURF_create()

# # find the keypoints and descriptors with SURF
# kp1, des1 = surf.detectAndCompute(img1,None)
# kp2, des2 = surf.detectAndCompute(img2,None)

# # BFMatcher with default params
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2, k=2)

# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
#         a=len(good)
#         percent=(a*100)/len(kp2)
#         print("{} % similarity".format(percent))
#         if percent >= 75.00:
#             print('Match Found')
#         if percent < 75.00:
#             print('Match not Found')

# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
# plt.imshow(img3),plt.show()