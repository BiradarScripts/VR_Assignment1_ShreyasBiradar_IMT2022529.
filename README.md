# Assignment 1: Visual Recognition

**Author:** Shreyas Biradar  
**ID:** IMT2022529  


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
![image](https://github.com/user-attachments/assets/5652d815-c5a7-47b5-9283-57950ea86950)

### Segmentation and Classification
Segmentation isolates **regions of interest (ROI)** in an image. Steps include:
- Load and resize the input image.
- Apply **Gaussian blur** to reduce noise.
- Convert to grayscale and apply **thresholding**.
- Detect contours and compute their areas.
- Assign unique colors for segmentation.

**Example: Segmented Coins**
![image](https://github.com/user-attachments/assets/8c9525d0-76e9-4ab2-aecc-313422e14d67)

### Final Count
Object detection and counting involve:
- Loading and resizing the image.
- Applying **Gaussian blur**.
- Converting to grayscale and applying **thresholding**.
- Detecting contours and filtering objects based on area.
- Visualizing detected objects.

**Example: Thresholded Coins**
![image](https://github.com/user-attachments/assets/7301e2de-9e10-4397-8845-f3b0e024cf64)


---

## Panoramic Image Stitching
### Feature Detection and Matching
Panoramic image stitching aligns multiple overlapping images into a seamless view. Key feature detection methods include:
- **SIFT** (Scale-Invariant Feature Transform)
- **ORB** (Oriented FAST and Rotated BRIEF)
- **SURF** (Speeded-Up Robust Features)

**Example: Stitched Panorama**
![image](https://github.com/user-attachments/assets/85eef055-b51c-4eb0-81c7-4fc7e52b9601)

**Example: Keypoints in an Image**
![image](https://github.com/user-attachments/assets/121dea88-f354-4f77-a1be-2759cb8b2239)

**Example: Matching Keypoints**
![image](https://github.com/user-attachments/assets/d8f35e57-37ac-43ae-99df-cf2aa168efee)

**Example: Final Keypoint Matching**
![image](https://github.com/user-attachments/assets/ca578aea-9c16-4b3b-86cc-d61e68e7e568)

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
## Folder Structuring
```bash
VisualRecognition-Assignment1-/
│
├── Part1                  
│   ├── coinDetect.py/             
│   ├── coinSegment.py/                
│   └── countCoins.py/
│
├── Part2                      
│   ├── panaroma.py/ 
│                     
|
├── img
│   ├── ...
│                                              
```
## Setup Instructions

Follow these steps to set up the project:

```bash
# Clone the repository
git clone https://github.com/BiradarScripts/VR_Assignment1_ShreyasBiradar_IMT2022529..git

# Navigate into the project directory
cd VisualRecognition-Assignment1-

# Create a new Conda environment with Python 3.10
conda create -p venv/ python==3.10

# Activate the environment
conda activate venv/

# Install required dependencies
pip install -r requirements.txt

# Navigate to Part1 directory
cd Part1

# Run the coin detection script
python coinDetect.py

# Run the coin segmentation script
python coinSegment.py

# Run the coin counting script
python countCoins.py

# Navigate back to the root directory
cd ..

# Navigate to Part2 directory
cd Part2

# Run the panorama stitching script
python panaroma.py
```

