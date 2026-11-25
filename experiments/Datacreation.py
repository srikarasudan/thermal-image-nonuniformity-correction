import cv2
import numpy as np

def rgb_to_gray(rgb_img):
    return cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

def create_bias_field(shape, amplitude=40, sigma=100, center=None):
    if center is None:
        center = (shape // 2, shape[1] // 2)
    Y, X = np.ogrid[:shape, :shape[1]]
    bias = amplitude * np.exp(-((X - center[1])**2 + (Y - center[0])**2) / (2 * sigma**2))
    return bias

def degrade_image(gray_img):
    bias_field = create_bias_field(gray_img.shape)
    noise = np.random.normal(0, 2, gray_img.shape)  # Optional: small gaussian noise
    degraded = gray_img.astype(np.float32) + bias_field + noise
    degraded = np.clip(degraded, 0, 255).astype(np.uint8)
    return degraded

# Load RGB image, then process
rgb_img = cv2.imread('input_rgb.jpg')
gray_img = rgb_to_gray(rgb_img)
degraded_img = degrade_image(gray_img)
cv2.imwrite('degraded_image.jpg', degraded_img)