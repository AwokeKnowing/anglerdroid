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
pts1 = np.argwhere(image01 > 0).astype(float)
pts2 = np.argwhere(image02 > 0).astype(float)

H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

# Extract the rotation and translation
rotation = np.arctan2(H[1, 0], H[0, 0]) * 180 / np.pi
translation_x = H[0, 2]
translation_y = H[1, 2]

print("Recovered Rotation: {:.2f} degrees".format(rotation))
print("Recovered Translation (x, y): ({:.2f}, {:.2f})".format(translation_x, translation_y))

cv2.waitKey(0)
cv2.destroyAllWindows()
