import cv2
import numpy as np
import time

# Create a black 1-channel image
image_size = (800, 800)
image = np.zeros(image_size, dtype=np.uint8)

# Define the color (255 for grayscale)
color = (255,)

# Define the shape properties
radius = 30

# Draw 1000 circles using OpenCV and measure the time
start_time = time.time()
for _ in range(500):
    center = (np.random.randint(0, image_size[1]), np.random.randint(0, image_size[0]))
    cv2.circle(image, center, radius, color, -1)  # -1 to fill the circle

circle_time_opencv = time.time() - start_time

# Reset the image
image = np.zeros(image_size, dtype=np.uint8)

# Draw 1000 squares using OpenCV and measure the time
start_time = time.time()
for _ in range(500):
    top_left = (np.random.randint(0, image_size[1]), np.random.randint(0, image_size[0]))
    bottom_right = (top_left[0] + 2 * radius, top_left[1] + 2 * radius)
    cv2.rectangle(image, top_left, bottom_right, color, -1)  # -1 to fill the square

square_time_opencv = time.time() - start_time

# Reset the image
#image = np.zeros(image_size, dtype=np.uint8)

# Draw 1000 squares using NumPy and measure the time
start_time = time.time()
#for _ in range(1000):
#    top_left = (np.random.randint(0, image_size[1]), np.random.randint(0, image_size[0]))
#    bottom_right = (top_left[0] + 2 * radius, top_left[1] + 2 * radius)
#    image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = color

square_time_numpy = time.time() - start_time

print(f"Time to draw 1000 circles using OpenCV: {circle_time_opencv:.6f} seconds")
print(f"Time to draw 1000 squares using OpenCV: {square_time_opencv:.6f} seconds")
print(f"Time to draw 1000 squares using NumPy: {square_time_numpy:.6f} seconds")

# Display the result (you may need to close the window to continue execution)
cv2.imshow("Image", image)
cv2.waitKey(5000)
cv2.destroyAllWindows()
