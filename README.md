# Assignment 1: Visual Recognition

**Author:** Shreyas Biradar  
**ID:** IMT2022529  
**Date:** $(date)

## Abstract
Computer vision is a vast and evolving field that enables machines to interpret and make decisions based on visual data. This report focuses on two major applications: **image stitching for panoramic generation** and **object detection using machine learning**. 

- The first section explores methodologies of **contour analysis** and **deep learning techniques** for object detection.  
- The second section elaborates on **image stitching**, including **homography computation** and **feature matching**.

Key challenges such as **occlusion, illumination variations, and feature selection** are discussed, along with possible enhancements to improve accuracy and performance.

---

## Introduction
Visual recognition is a key component of computer vision, enabling applications such as **autonomous navigation, medical diagnostics, and surveillance**. Object detection and image stitching are fundamental techniques used in creating immersive panoramas and enabling intelligent scene understanding.

## Object Detection and Analysis
### Detection Methodology
Object detection involves multiple approaches, from **traditional image processing techniques** to **deep learning-based methods**. The fundamental pipeline includes:
1. Convert the input image to **grayscale**.
2. Apply **adaptive thresholding** or **edge detection**.
3. Use **contour detection** to extract object shapes.
4. Apply **filtering techniques** to remove noise.
5. Use a **classifier (deep learning model)** to categorize objects.

**Example: Detected Coins**
![Detected Coins](coinsDetect.png)

### Segmentation and Classification
Segmentation isolates **regions of interest (ROI)** in an image. Steps include:
- Load and resize the input image.
- Apply **Gaussian blur** to reduce noise.
- Convert to grayscale and apply **thresholding**.
- Detect contours and compute their areas.
- Assign unique colors for segmentation.

**Example: Segmented Coins**
![Segmented Coins](coinsSegment.png)

### Final Count
Object detection and counting involve:
- Loading and resizing the image.
- Applying **Gaussian blur**.
- Converting to grayscale and applying **thresholding**.
- Detecting contours and filtering objects based on area.
- Visualizing detected objects.

**Example: Thresholded Coins**
![Thresholded Coins](thresholded.png)

---

## Panoramic Image Stitching
### Feature Detection and Matching
Panoramic image stitching aligns multiple overlapping images into a seamless view. Key feature detection methods include:
- **SIFT** (Scale-Invariant Feature Transform)
- **ORB** (Oriented FAST and Rotated BRIEF)
- **SURF** (Speeded-Up Robust Features)

**Example: Stitched Panorama**
![Stitched Result](stitched_result.jpg)

**Example: Keypoints in an Image**
![Keypoints](keypoints.png)

**Example: Matching Keypoints**
![Matching Keypoints](matching_keypoints.png)

**Example: Final Keypoint Matching**
![Final Matching](inliers_outliers.png)

### Homography Estimation with RANSAC
- Uses **RANSAC (Random Sample Consensus)** to remove outliers.
- Computes a **homography matrix** for aligning images.
- Visualizes inliers (green) and outliers (red).

### Image Warping and Stitching
- Applies a **perspective transformation** using the computed homography matrix.
- Uses **stitching techniques** to blend the images.

---

## Challenges and Solutions
### 1. Illumination Variations
- Affects segmentation and feature extraction.
- **Solution:** Use **adaptive histogram equalization** and **normalization techniques**.

### 2. Occlusions and Noise
- Reduces segmentation accuracy.
- **Solution:** Use **deep learning models (YOLO, Faster R-CNN)** and **Gaussian blurring**.

### 3. Feature Matching Issues
- False matches affect recognition.
- **Solution:** Use **ORB** for efficiency.

### 4. Threshold Selection Sensitivity
- Choosing the right threshold is challenging.
- **Solution:** Use **adaptive thresholding** or **Otsu’s method**.

### 5. Contour Detection and Overlapping Objects
- Similar intensities may merge objects.
- **Solution:** Use **hierarchical contour analysis** or **watershed segmentation**.

---

## Conclusion
This report demonstrated object detection and panoramic image stitching techniques. **Future work** will focus on:
- Improving segmentation accuracy.
- Refining feature extraction methods.
- Exploring **deep learning architectures** for real-time applications.

---

## References
- OpenCV Documentation: [https://docs.opencv.org/](https://docs.opencv.org/)
- Deep Learning for Object Detection: [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)
- Image Stitching Techniques: [https://www.cs.cornell.edu/courses/cs4670/2018fa/](https://www.cs.cornell.edu/courses/cs4670/2018fa/)
