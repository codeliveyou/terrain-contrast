import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

def show_image(image):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.colorbar()
    plt.show()

def show_mask(mask):
    if mask is not None:
        cv2.imshow("Mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def fourier_mellin_transform(image, mask = None):
    # Convert to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if mask is not None:
        # show_image(image_gray)
        image_gray = cv2.bitwise_and(image_gray, image_gray, mask=mask)
        # show_image(image_gray)

    # Log-Polar Transform
    flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG
    log_polar_image = cv2.warpPolar(image_gray, (image_gray.shape[0], image_gray.shape[1]), 
                                    (image_gray.shape[0]//2, image_gray.shape[1]//2), 
                                    60, flags)
    # show_image(log_polar_image)
    
    log_polar_image = log_polar_image.astype(np.float32)
    log_polar_image -= log_polar_image.min()
    log_polar_image /= log_polar_image.max()
    
    f = np.fft.fft2(log_polar_image)
    return f

def create_mask(image):
    # Assuming erased parts are black or a specific color, adjust the threshold accordingly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)  # Threshold needs to be adjusted based on the image
    return mask


def compare_images_fft(imageA, imageB, mask = None):
    
    fA = fourier_mellin_transform(imageA)
    fB = fourier_mellin_transform(imageB, mask)

    
    fA /= np.std(fA).sum()
    fB /= np.std(fB).sum()

    
    fA_shifted = np.fft.fftshift(fA)
    fB_shifted = np.fft.fftshift(fB)
    cross_correlation = np.abs(np.fft.ifft2(np.conj(fA_shifted) * fB_shifted))

    max_correlation = np.max(cross_correlation)
    return max_correlation




def calculate_rotation(imageA, imageB, mask = None):
    fA = fourier_mellin_transform(imageA)
    fB = fourier_mellin_transform(imageB, mask)

    fA /= np.std(fA).sum()
    fB /= np.std(fB).sum()

    cross_correlation = np.abs(np.fft.ifft2(np.conj(fA) * fB))
    
    max_idx = np.unravel_index(np.argmax(cross_correlation), cross_correlation.shape)
    
    angle_per_pixel = 360 / cross_correlation.shape[0]
    rotation_angle = (max_idx[0] * angle_per_pixel) % 360
    if rotation_angle > 180:
        rotation_angle -= 360

    return rotation_angle

def add_erased_part(image):
    height = image.shape[0]
    width = image.shape[1]
    random.seed(513)
    for _ in range(5):
        h = random.randint(5, 25)
        w = random.randint(5, 45)
        x = random.randint(0, height - h)
        y = random.randint(0, width - w)
        image[x : x + h, y : y + w] = 255
    return image

def max_correlation(imageA, imageB, mask = None):
    max_score = 0
    height_A = imageA.shape[0]
    width_A = imageA.shape[1]

    height_B = imageB.shape[0]
    width_B = imageB.shape[1]

    # show_image(imageB)

    # imageB = cv2.multiply(imageB, 2)

    target_X = 0
    target_Y = 0

    for coner_X in range(0, height_A - height_B + 1, 5):
        for coner_Y in range(0, width_A - width_B + 1, 5):
            current_correlation = compare_images_fft(imageA[coner_X : coner_X + height_B, coner_Y : coner_Y + width_B], imageB, mask)
            # print(current_correlation)
            # print(current_correlation)
            if current_correlation > max_score:
                target_X = coner_X
                target_Y = coner_Y
                max_score = current_correlation
    # show_image(imageA[target_X : target_X + height_B, target_Y : target_Y + width_B])
    angle = calculate_rotation(imageA[target_X : target_X + height_B, target_Y : target_Y + width_B], imageB)


    for x in range(target_X, target_X + height_B):
        imageA[x][target_Y] = 0
        imageA[x][target_Y + height_B] = 0
    for y in range(target_Y, target_Y + width_B):
        imageA[target_X][y] = 0
        imageA[target_X + height_B][y] = 0

    show_image(imageA) #[target_X : target_X + height_B, target_Y : target_Y + width_B])
    return [max_score, target_X, target_Y, angle]

# Load images
imageA = cv2.imread('./images/A.jpg')
imageB = cv2.imread('./images/C.jpg')

# imageB = add_erased_part(imageB)

mask = create_mask(imageB)
# mask = None
# show_mask(mask)


print(max_correlation(imageA, imageB, mask))


