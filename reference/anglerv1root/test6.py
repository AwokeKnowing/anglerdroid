import cv2
import numpy as np
import random

# Generate a 240x224 black image (image01)
image01 = np.zeros((224, 240), dtype=np.uint8)

# Create 5 random white squares on image01
for _ in range(5):
    x = random.randint(0, 240 - 50)
    y = random.randint(0, 224 - 50)
    cv2.rectangle(image01, (x, y), (x + 50, y + 50), (255,), -1)

# Copy image01 to image02
image02 = image01.copy()

# Rotate image02 by 5 degrees and translate it 5 pixels up and 7 pixels left
rows, cols = image02.shape
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 5, 1)
M[0, 2] -= 7
M[1, 2] -= 5
image02 = cv2.warpAffine(image02, M, (cols, rows))

# Show both images
cv2.imshow("image01", image01)
cv2.imshow("image02", image02)

# Find the homography transformation between image01 and image02
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
kp1, des1 = sift.detectAndCompute(image01, None)
kp2, des2 = sift.detectAndCompute(image02, None)

bf = cv2.BFMatcher()

# Perform KNN matching with k=6
k = 6
matches = bf.knnMatch(des1, des2, k=k)

# Apply a more stringent distance ratio test
good = []
for m in matches:
    # Sort the matches by distance
    m = sorted(m, key=lambda x: x.distance)

    # Apply the distance ratio test to the top two closest matches
    if m[0].distance < 0.6 * m[1].distance:
        good.append(m[0])

# Ensure there are enough good matches
if len(good) >= 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Calculate the homography
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Extract the rotation and translation
    rotation = np.arctan2(H[1, 0], H[0, 0]) * 180 / np.pi
    translation_x = H[0, 2]
    translation_y = H[1, 2]

    print("Recovered Rotation: {:.2f} degrees".format(rotation))
    print("Recovered Translation (x, y): ({:.2f}, {:.2f})".format(translation_x, translation_y))
else:
    print("Not enough good matches to estimate homography.")
exit()

























# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

# Ensure there are enough good matches
if len(good) >= 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Calculate the homography
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Extract the rotation and translation
    rotation = np.arctan2(H[1, 0], H[0, 0]) * 180 / np.pi
    translation_x = H[0, 2]
    translation_y = H[1, 2]

    print("Recovered Rotation: {:.2f} degrees".format(rotation))
    print("Recovered Translation (x, y): ({:.2f}, {:.2f})".format(translation_x, translation_y))
else:
    print("Not enough good matches to estimate homography.")

cv2.waitKey(0)
cv2.destroyAllWindows()
