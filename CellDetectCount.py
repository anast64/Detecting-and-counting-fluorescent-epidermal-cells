"""
Cell Detect count takes in images from a folder and performs watershed
to separate cells then maps contours to foreground objects. Foreground
objects have area, perimeter, and circularity extracted from them to 
compare with set thresholds based on distribution of features from skin
cells. Only the contours with features that match skin cells will be added
to cell count, and the printed output will be the cell count for that image.
This process is repeated for all images provided in the folder.
"""
import cv2
import numpy as np
import pandas as pd
import os

# Define watershed function to separate clumped cells
def watershed(image):
    # convert input image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding to create binary image
    ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # Finding the sure background area by dilating foreground objects.
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(thresh, kernel, iterations=3)

    # Finding sure foreground area using distance transform and thresholding it
    dist_trans = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_trans, 0.05 * dist_trans.max(), 255, 0)

    # Find unknown region by subtracting sure foreground from sure background
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label markers for the connected components in the sure foreground
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1  # Increment labels by 1 so that sure_bg is 1
    markers[unknown == 255] = 0  # Mark the region of unknown with 0

    # Apply watershed algorithm to segment the objects in the image
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]  # Mark watershed boundaries with red color

    # Find contours from the sure foreground and return them
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# get the path/directory of folder with images of cells
folder_dir = "C:/Users/anast/Documents/Semester 1 Hns/ENGR7761 Computer Vision/Detect_and_count/LT"

# Complete this for every image within the folder
for image_file in os.listdir(folder_dir):
    # Load the image
    img_path = os.path.join(folder_dir, image_file)
    img = cv2.imread(img_path)

    # Check if image is read properly or exists in folder
    if img is None:
        print(f"Failed to load {img_path}")
        continue

    # Create copy of original image
    img_copy = img.copy()
    # Apply watershed algorithm to the copied image and find objects
    contours = watershed(img_copy)

    # Initialize lists to store features and cell contours
    features = []
    cell_contours = []

    # Extract features for all contours detected in image
    for cnt in contours:
        area = cv2.contourArea(cnt) # Calc area of contour
        perimeter = cv2.arcLength(cnt, True) # Calc perimeter of contour
        perimeter = round(perimeter, 4) # Round perimeter to 4 decimal places

        # Calc circularity of contour
        if perimeter == 0:
            # if zero assume it's a dot (perfect circle)
            circularity = 1
        else:
            circularity = (4 * np.pi * area) / (perimeter ** 2)

        # Append features to respective arrays if they meet criteria
        if (area <= 40):
            if (perimeter <= 45):
                if (circularity >= 0.4):
                    cell_contours.append(cnt) # Add contour to the list of cell contours

                    # Create feature vector with area, perimeter, and circularity
                    feature_vector = [area, perimeter, circularity]
                    features.append(feature_vector) # Add feature vector to features list

    # Print cell count for the current image file
    print(f"The cell count for {image_file} is: {len(cell_contours)}")
