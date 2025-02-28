import numpy as np
import matplotlib.pyplot as plt
import cv2

def display_images(img_list, titles, save_path=None):
    plt.figure(figsize=(12, 6))
    for idx, (image, title) in enumerate(zip(img_list, titles)):
        plt.subplot(1, len(img_list), idx + 1)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

img_path = "./../img/three.jpg"

# Load and preprocess the image
original_img = cv2.imread(img_path)
resized_img = cv2.resize(original_img, (720, 900))
processed_img = resized_img.copy()

display_images([cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)], ["Original Image"])

# Apply Gaussian blur
blurred_img = cv2.GaussianBlur(resized_img, (7, 7), 10)
display_images([cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB)], 
               ["Original", "Blurred"])

# Convert to grayscale
gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
display_images([cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB), gray_img], 
               ["Blurred", "Grayscale"])

# Apply threshold
_, binary_img = cv2.threshold(gray_img, 170, 255, cv2.THRESH_BINARY)
display_images([gray_img, binary_img], ["Grayscale", "Thresholded"], "thresholded.png")

# Find contours
contours, _ = cv2.findContours(binary_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contour_areas = {idx: cv2.contourArea(cnt) for idx, cnt in enumerate(contours)}
sorted_areas = sorted(contour_areas.items(), key=lambda x: x[1], reverse=True)
detected_objects = np.array(sorted_areas).astype(int)
valid_contours = np.argwhere(detected_objects[:, 1] > 500).shape[0]

# Draw contours
for i in range(1, valid_contours):
    processed_img = cv2.drawContours(processed_img, contours, detected_objects[i, 0], (0, 255, 0), 3)

display_images([cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)], 
               ["Original", "Detected Objects"], "coinsDetect.png")
