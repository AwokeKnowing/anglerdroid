import cv2
import numpy as np
import random
import math
import time

image = cv2.imread('floorexample.png',cv2.IMREAD_GRAYSCALE)
# Create a 240x224 black image (image01) with a 64-pixel black margin
image01 = np.zeros((224, 240), dtype=np.uint8)
image01[64:160, 64:176] = 255  # Add a white square within the margin

# Copy image01 to image02
image02 = image01.copy()

# Rotate image02 by 5 degrees and translate it 5 pixels up and 7 pixels left
rows, cols = image02.shape
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 5, 1)
M[0, 2] -= 7
M[1, 2] -= 5
image02 = cv2.warpAffine(image02, M, (cols, rows))

# Create scaled-down copies of image01 and image02
scaling_factor = 1/1
img1 = cv2.resize(image01, None, fx=scaling_factor, fy=scaling_factor)
img2s = cv2.resize(image02, None, fx=scaling_factor, fy=scaling_factor)

# Define parameters for the search
rotation_range = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-2,0,1,2,3,4,5,6,7,8,9,10]
translation_range = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-2,0,1,2,3,4,5,6,7,8,9,10]

# Create a list to store the results
results = []

start = time.time()
try:
    for rotation_angle in rotation_range:
        # Rotate img2
        img2 = img2s.copy()
        M_rotation = cv2.getRotationMatrix2D((img2.shape[1] / 2, img2.shape[0] / 2), rotation_angle, 1)
        
        img2_rotated = cv2.warpAffine(img2, M_rotation, (img2.shape[1], img2.shape[0]))

    
        for dx in translation_range:
            for dy in translation_range:
                # Translate img2_rotated
                M_translation = np.float32([[1, 0, dx], [0, 1, dy]])
                img2_transformed = cv2.warpAffine(img2_rotated, M_translation, (img2.shape[1], img2.shape[0]))

                diff = cv2.absdiff(img1, img2_transformed)
                non_zero_pixels = np.sum(diff[diff == 255])
                
                print(rotation_angle,dx,dy,non_zero_pixels)
                results.append((rotation_angle, (dx, dy), non_zero_pixels))
                if non_zero_pixels < 200:
                    raise Exception
except Exception:
    pass

print(time.time() - start)
# Sort the results based on the number of non-zero pixels
sorted_results = sorted(results, key=lambda x: x[2])

# Show the top 10 results
for i, (rotation_angle, (dx, dy), non_zero_pixels) in enumerate(sorted_results[:10], 1):
    dx = dx/scaling_factor
    dy = dy/scaling_factor
    print(f"Top {i} - Rotation: {rotation_angle}°, Translation: ({dx}, {dy}), Non-zero pixels: {non_zero_pixels}")
