import timeit
import math
import numpy as np

a = 3.0
b = 4.0

# Measure the execution time of math.sqrt
math_sqrt_time = timeit.timeit('math.sin(a**2 + b**2)', globals=globals(), number=1000000)
print(f"Time taken by math.sqrt: {math_sqrt_time:.6f} seconds")

# Measure the execution time of np.linalg.norm
np_linalg_norm_time = timeit.timeit('np.sin(a**2+b**2)', globals=globals(), setup='import numpy as np', number=1000000)
print(f"Time taken by np.linalg.norm: {np_linalg_norm_time:.6f} seconds")

# Calculate and print the speed difference
speed_difference = math_sqrt_time / np_linalg_norm_time
print(f"np.linalg.norm is {speed_difference:.2f} times faster than math.sqrt for finding the hypotenuse.")
