import timeit
import numpy as np

# Define the two points as tuples
pointa = (2, 3)
pointb = (5, 7)

# Calculate the distance by subtracting tuples
def distance_tuple_subtraction(pointa, pointb):
    return ((pointb[0] - pointa[0])**2 + (pointb[1] - pointa[1])**2)**0.5

# Convert tuples to arrays and calculate the distance
def distance_array_conversion(pointa, pointb):
    pointa_array = np.array(pointa)
    pointb_array = np.array(pointb)
    return np.linalg.norm(pointb_array - pointa_array)

# Time the distance calculation with tuple subtraction
tuple_subtraction_time = timeit.timeit(lambda: distance_tuple_subtraction(pointa, pointb), number=1000000)

# Time the distance calculation with array conversion
array_conversion_time = timeit.timeit(lambda: distance_array_conversion(pointa, pointb), number=1000000)

print(f"Time for tuple subtraction: {tuple_subtraction_time:.6f} seconds")
print(f"Time for array conversion: {array_conversion_time:.6f} seconds")
