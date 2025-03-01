import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def show_plot(fig, save_path=False):
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()

image_1 = cv2.imread('./../img/leftp.jpeg')  
image_2 = cv2.imread('./../img/rightp.jpeg')  

gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

feature_detector = cv2.ORB_create()
kp_1, des_1 = feature_detector.detectAndCompute(gray_1, None)
kp_2, des_2 = feature_detector.detectAndCompute(gray_2, None)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(cv2.drawKeypoints(image_1, kp_1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
ax[0].set_title("Keypoints in First Image")
ax[0].axis('off')
ax[1].imshow(cv2.drawKeypoints(image_2, kp_2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
ax[1].set_title("Keypoints in Second Image")
ax[1].axis('off')
show_plot(fig, save_path="keypoints.png")

bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf_matcher.match(des_1, des_2)
matches = sorted(matches, key=lambda x: x.distance)
match_points = [[kp_1[m.queryIdx].pt[0], kp_1[m.queryIdx].pt[1], kp_2[m.trainIdx].pt[0], kp_2[m.trainIdx].pt[1]] for m in matches]

def get_homography(match_pairs):
    A_matrix = []
    for p in match_pairs:
        x, y = p[0], p[1]
        X, Y = p[2], p[3]
        A_matrix.append([x, y, 1, 0, 0, 0, -X*x, -X*y, -X])
        A_matrix.append([0, 0, 0, x, y, 1, -Y*x, -Y*y, -Y])
    A_matrix = np.array(A_matrix)
    _, _, vh = np.linalg.svd(A_matrix)
    homography_matrix = vh[-1, :].reshape(3, 3)
    homography_matrix /= homography_matrix[2, 2]
    return homography_matrix

homography = get_homography(match_points[:10])  
print("Computed Homography Matrix:")
print(homography)

match_display = cv2.drawMatches(image_1, kp_1, image_2, kp_2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(12, 6))
plt.imshow(match_display)
plt.title("Matching Keypoints")
plt.axis('off')
show_plot(plt.gcf(), save_path="matchedKeypoints.png")

def ransac_homography(data_points, threshold=5, iterations=5000):
    best_inliers = []
    optimal_homography = None
    for _ in range(iterations):
        subset = random.choices(data_points, k=4)
        H_matrix = get_homography(subset)
        inliers = []
        for p in data_points:
            src_pt = np.array([p[0], p[1], 1]).reshape(3, 1)
            dst_pt = np.array([p[2], p[3], 1]).reshape(3, 1)
            mapped_pt = np.dot(H_matrix, src_pt)
            mapped_pt /= mapped_pt[2]
            error = np.linalg.norm(dst_pt - mapped_pt)
            if error < threshold:
                inliers.append(p)
        if len(inliers) > len(best_inliers):
            best_inliers, optimal_homography = inliers, H_matrix
    return optimal_homography, best_inliers

final_homography, refined_matches = ransac_homography(match_points)

outlier_points = [p for p in match_points if p not in refined_matches]

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
for p in refined_matches:
    ax.plot([p[0], p[2]], [p[1], p[3]], 'g-', alpha=0.6)
for p in outlier_points:
    ax.plot([p[0], p[2]], [p[1], p[3]], 'r-', alpha=0.3)
ax.set_title("Inliers (Green) vs Outliers (Red)")
ax.axis('off')
show_plot(fig, save_path="Liners.png")

h_1, w_1 = image_2.shape[:2]
h_2, w_2 = image_1.shape[:2]
pts_1 = np.float32([[0, 0], [0, h_1], [w_1, h_1], [w_1, 0]]).reshape(-1, 1, 2)
pts_2 = np.float32([[0, 0], [0, h_2], [w_2, h_2], [w_2, 0]]).reshape(-1, 1, 2)
transformed_pts = cv2.perspectiveTransform(pts_2, final_homography)
full_pts = np.concatenate((pts_1, transformed_pts), axis=0)
[x_min, y_min] = np.int32(full_pts.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(full_pts.max(axis=0).ravel() + 0.5)
H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]).dot(final_homography)
stitched_image = cv2.warpPerspective(image_1, H_translation, (x_max - x_min, y_max - y_min))
stitched_image[-y_min:h_1 + (-y_min), -x_min:w_1 + (-x_min)] = image_2
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
plt.title("Final Stitched Image")
plt.axis('off')
show_plot(plt.gcf(), save_path="Panaroma.png")
cv2.imwrite('stichedResult.jpg', stitched_image)
print("Panorama image saved successfully.")